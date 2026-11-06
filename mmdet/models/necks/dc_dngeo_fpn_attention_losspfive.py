"""
dc_dngeo_fpn_attention_losspfive.py

合并了两个创新点的 FPN 版本：
1. 来自 `dngeo_fpn_attention_losspfive.py`:
   - PositionDenoisingModule: 在 1×1 lateral 之后、top-down 之前进行位置去噪
   - FeatureEncoder + geo_loss: 几何对比学习损失（仅在部分尺度上，默认 levels=[2:]）

2. 来自 `dc_fpn.py`:
   - _DCNAlign: 在 FPN top-down 上采样融合处使用可变形卷积（DCNv2）进行特征对齐
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, constant_init, normal_init
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d
from mmcv.runner import auto_fp16

from ..builder import NECKS
from .fpn import FPN


class FeatureEncoder(nn.Module):
    """特征编码器：将特征图编码为向量表示，用于对比学习."""

    def __init__(self, in_channels=256, hidden_dim=256):
        super().__init__()
        self.E = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        fea = self.E(x).squeeze(-1).squeeze(-1)  # [B, C]
        return self.mlp(fea)  # [B, D]


class PositionDenoisingModule(nn.Module):
    """位置去噪模块：专注于去除 1×1 lateral 卷积引入的位置噪声.

    该实现来自 `mmdet/models/necks/dngeo_fpn.py`，用于对 lateral 特征做轻量去噪。
    """

    def __init__(self, in_channels=256, num_scales=4):
        super().__init__()
        self.num_scales = num_scales

        self.denoising_convs = nn.ModuleList(
            [
                ConvModule(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=dict(type="BN"),
                    act_cfg=dict(type="ReLU", inplace=True),
                )
                for _ in range(num_scales)
            ]
        )

        self.spatial_attentions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),
                    nn.Sigmoid(),
                )
                for _ in range(num_scales)
            ]
        )

        self.channel_attentions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(in_channels, in_channels // 16, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels // 16, in_channels, 1),
                    nn.Sigmoid(),
                )
                for _ in range(num_scales)
            ]
        )

        self.denoising_weights = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.5)) for _ in range(num_scales)]
        )

    def forward(self, lateral_features, backbone_features=None):
        denoised_features = []
        for i, lateral_feat in enumerate(lateral_features):
            denoised = self.denoising_convs[i](lateral_feat)

            spatial_att = self.spatial_attentions[i](denoised)
            denoised = denoised * spatial_att

            channel_att = self.channel_attentions[i](denoised)
            denoised = denoised * channel_att

            weight = torch.sigmoid(self.denoising_weights[i])
            denoised_feat = weight * denoised + (1 - weight) * lateral_feat
            denoised_features.append(denoised_feat)

        return denoised_features


class _DCNAlign(nn.Module):
    """在 FPN top-down 上采样融合处做特征对齐。

    给定同分辨率的 (p_i, up(p_{i+1}))，从二者 concat 的特征预测 DCNv2 的 offset+mask，
    然后对 up(p_{i+1}) 施加可变形卷积，得到与 p_i 对齐后的特征。
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        deform_groups: int = 1,
        init_zero_offset: bool = True,
    ):
        super().__init__()
        assert kernel_size in (1, 3), '当前实现建议 kernel_size=1 或 3'
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.deform_groups = int(deform_groups)
        self.init_zero_offset = bool(init_zero_offset)

        k2 = self.kernel_size * self.kernel_size
        # 每个采样点 2 个偏移、1 个掩码
        self.offset_dim = 2 * k2 * self.deform_groups
        self.mask_dim = 1 * k2 * self.deform_groups
        self.offset_and_mask_dim = self.offset_dim + self.mask_dim

        # 用 (p_i, up(p_{i+1})) 的 concat 预测 offset+mask
        self.offset_and_mask = nn.Conv2d(
            self.channels * 2,
            self.offset_and_mask_dim,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            bias=True,
        )

        # 对 up(p_{i+1}) 做 DCNv2（输出通道不变）
        self.dcn = ModulatedDeformConv2d(
            self.channels,
            self.channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            deform_groups=self.deform_groups,
            bias=True,
        )

        self._init_weights()

    def _init_weights(self):
        # 保持与仓库内 DyHead 的初始化习惯一致
        normal_init(self.dcn, mean=0, std=0.01)
        if self.init_zero_offset:
            constant_init(self.offset_and_mask, 0)
        else:
            normal_init(self.offset_and_mask, mean=0, std=0.01)

    def forward(self, p_i, p_ip1_up):
        # p_i / p_ip1_up: [B, C, H, W]
        offset_and_mask = self.offset_and_mask(torch.cat([p_i, p_ip1_up], dim=1))
        offset = offset_and_mask[:, : self.offset_dim, :, :]
        mask = offset_and_mask[:, self.offset_dim :, :, :].sigmoid()
        aligned = self.dcn(p_ip1_up.contiguous(), offset, mask)
        return aligned


@NECKS.register_module()
class DC_DNGeo_FPNAttention_LossPfive(FPN):
    """合并了位置去噪和可变形卷积对齐的 DNGeo FPN 版本.

    核心创新点：
    1. PositionDenoisingModule: 在 lateral 特征上进行位置去噪
    2. _DCNAlign: 在 top-down 融合处使用可变形卷积对齐
    3. geo_loss: 几何对比学习损失（仅在部分尺度上，默认 levels=[2:]）
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        start_level=0,
        end_level=-1,
        add_extra_convs=False,
        relu_before_extra_convs=False,
        no_norm_on_lateral=False,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
        upsample_cfg=dict(mode="nearest"),
        # PositionDenoisingModule 相关
        use_denoising=True,
        denoising_weight=1.0,
        # _DCNAlign 相关
        dcn_kernel_size: int = 3,
        dcn_deform_groups: int = 1,
        dcn_zero_init_offset: bool = True,
        # geo_loss 相关
        return_loss=True,
        contrastive_temp=0.5,
        init_cfg=dict(type="Xavier", layer="Conv2d", distribution="uniform"),
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            start_level=start_level,
            end_level=end_level,
            add_extra_convs=add_extra_convs,
            relu_before_extra_convs=relu_before_extra_convs,
            no_norm_on_lateral=no_norm_on_lateral,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            upsample_cfg=upsample_cfg,
            init_cfg=init_cfg,
        )

        self.use_denoising = use_denoising
        self.denoising_weight = denoising_weight
        self.return_loss = return_loss
        self.contrastive_temp = contrastive_temp

        # PositionDenoisingModule: 在 lateral 之后进行去噪
        if self.use_denoising:
            num_scales = len(self.lateral_convs)
            self.position_denoising_module = PositionDenoisingModule(
                in_channels=out_channels, num_scales=num_scales
            )

        # _DCNAlign: 在 top-down 融合处进行对齐
        used_backbone_levels = self.backbone_end_level - self.start_level
        self.align_modules = nn.ModuleList()
        for _ in range(used_backbone_levels - 1):
            self.align_modules.append(
                _DCNAlign(
                    channels=out_channels,
                    kernel_size=dcn_kernel_size,
                    deform_groups=dcn_deform_groups,
                    init_zero_offset=dcn_zero_init_offset,
                )
            )

        # FeatureEncoder: 用于 geo_loss 计算
        if return_loss:
            self.lateral_loc_encoder = FeatureEncoder(out_channels)
            self.backbone_loc_encoder = FeatureEncoder(out_channels)

            used_in_channels = in_channels[self.start_level : self.backbone_end_level]
            self.channel_transfer = nn.ModuleList()
            for c in used_in_channels:
                if c == out_channels:
                    self.channel_transfer.append(nn.Identity())
                else:
                    self.channel_transfer.append(nn.Conv2d(c, out_channels, kernel_size=1))

    def _build_laterals_with_denoising(self, inputs):
        """构建 lateral 特征并进行去噪."""
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        if self.use_denoising:
            denoised_laterals = self.position_denoising_module(laterals)
            laterals = [
                orig + self.denoising_weight * (denoised - orig)
                for orig, denoised in zip(laterals, denoised_laterals)
            ]

        return laterals

    def _build_topdown_and_outputs(self, laterals, inputs):
        """构建 top-down 路径（带 DCN 对齐）并生成输出."""
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            if "scale_factor" in self.upsample_cfg:
                up = F.interpolate(laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                up = F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)

            # 使用 DCN 对齐：对 up(p_{i+1}) 进行可变形卷积对齐
            aligned_up = self.align_modules[i - 1](laterals[i - 1], up)
            laterals[i - 1] = laterals[i - 1] + aligned_up

        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]

        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for _ in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                if self.add_extra_convs == "on_input":
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == "on_lateral":
                    extra_source = laterals[-1]
                elif self.add_extra_convs == "on_output":
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        return tuple(outs)

    @auto_fp16()
    def forward(self, inputs, backbone_features=None):
        assert len(inputs) == len(self.in_channels)

        # 1. 构建 lateral 特征并进行去噪
        laterals = self._build_laterals_with_denoising(inputs)

        # 2. 构建 top-down 路径（带 DCN 对齐）并生成输出
        fpn_outs = self._build_topdown_and_outputs(laterals, inputs)

        # 3. 计算 geo_loss（如果训练且需要返回损失）
        if self.training and self.return_loss and backbone_features is not None:
            losses = self._compute_geo_loss(laterals, backbone_features)
            return fpn_outs, losses

        return fpn_outs

    def _compute_geo_loss(self, lateral_features, backbone_features):
        """计算几何对比学习损失（仅在部分尺度上，默认 levels=[2:]）."""
        x_b = backbone_features[self.start_level : self.backbone_end_level]
        x_lateral = list(lateral_features)

        n = min(len(x_b), len(x_lateral))
        x_b = x_b[:n]
        x_lateral = x_lateral[:n]

        if len(x_lateral) == 0:
            device = (
                lateral_features[0].device
                if len(lateral_features) > 0
                else torch.device("cpu")
            )
            return {"geo_loss": torch.tensor(0.0, device=device)}

        # 参与 geo_loss 的尺度索引（当前 pure_pfive 使用 [2:]）
        # 如果x_lateral长度小于3，则使用最后一个尺度
        if len(x_lateral) >= 3:
            levels = list(range(len(x_lateral)))[2:]
        else:
            # 如果只有2个或更少的尺度，使用最后一个尺度
            levels = [len(x_lateral) - 1] if len(x_lateral) > 0 else []
        
        if len(levels) == 0:
            device = x_lateral[0].device
            return {"geo_loss": torch.tensor(0.0, device=device)}

        batch_size = x_lateral[0].shape[0]
        device = x_lateral[0].device

        lateral_loc, backbone_loc = [], []
        for lvl in levels:
            lateral_loc.append(self.lateral_loc_encoder(x_lateral[lvl]))  # [B, D]
            backbone_aligned = self.channel_transfer[lvl](x_b[lvl])
            backbone_loc.append(self.backbone_loc_encoder(backbone_aligned))  # [B, D]

        loc_lateral_all = torch.stack(lateral_loc, dim=0).view(-1, lateral_loc[0].shape[-1])
        loc_backbone_all = torch.stack(backbone_loc, dim=0).view(-1, backbone_loc[0].shape[-1])

        loc_lateral_norm = F.normalize(loc_lateral_all, p=2, dim=1)
        loc_backbone_norm = F.normalize(loc_backbone_all, p=2, dim=1)

        sim_lateral_lateral = (
            torch.matmul(loc_lateral_norm, loc_lateral_norm.t()) / self.contrastive_temp
        )
        sim_lateral_backbone = (
            torch.matmul(loc_lateral_norm, loc_backbone_norm.t()) / self.contrastive_temp
        )

        scale_size = len(levels)
        scale_indices = torch.arange(scale_size, device=device).repeat_interleave(batch_size)
        batch_indices = torch.arange(batch_size, device=device).repeat(scale_size)

        scale_mask = scale_indices.unsqueeze(1) == scale_indices.unsqueeze(0)
        batch_mask = batch_indices.unsqueeze(1) == batch_indices.unsqueeze(0)
        pos_mask = scale_mask & batch_mask

        # 与 pure_pfive 一致：仅把"尺度不同 且 batch 不同"的样本当作负样本
        neg_mask = (~scale_mask) & (~batch_mask)
        
        # 如果neg_mask全为False（例如只有一个scale），则使用所有非正样本作为负样本
        if neg_mask.sum() == 0:
            # 使用所有非正样本作为负样本（包括不同batch的相同scale）
            neg_mask = ~pos_mask
            # 确保对角线（自己）不被当作负样本
            neg_mask = neg_mask.fill_diagonal_(False)

        exp_sim_lateral_backbone = torch.exp(sim_lateral_backbone)
        pos_sim = exp_sim_lateral_backbone * pos_mask
        numerators = pos_sim.sum(dim=1)

        exp_sim_lateral_lateral = torch.exp(sim_lateral_lateral)
        neg_sim = (exp_sim_lateral_lateral + exp_sim_lateral_backbone) * neg_mask
        denominators = neg_sim.sum(dim=1) + numerators
        
        # 如果denominators为0或nan，则使用lateral-lateral的相似度作为fallback
        # 这确保了即使在没有负样本的情况下，loss也能正常计算
        if denominators.sum() == 0 or torch.isnan(denominators).any():
            # 使用lateral-lateral的相似度作为额外的负样本项
            denominators = numerators + exp_sim_lateral_lateral.sum(dim=1)

        geo_loss = -torch.log(numerators / (denominators + 1e-8) + 1e-8).mean()
        return {"geo_loss": geo_loss}


# 兼容可能的 type 命名习惯（可选别名）
DC_DNGeo_FPN_Attention_LossPfive = DC_DNGeo_FPNAttention_LossPfive
