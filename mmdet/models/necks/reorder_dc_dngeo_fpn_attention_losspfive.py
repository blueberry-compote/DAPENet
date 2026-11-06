"""
reorder_dc_dngeo_fpn_attention_losspfive.py

方案 3：先 top-down 再在部分层去噪。

流程：laterals → top-down(DCN) → 仅对 denoising_levels 去噪 → fpn_convs → outs
可试 denoising_levels=[0, 1]（默认）。

dcn_levels：仅在部分 top-down 融合步使用 DCN（类比 dc_last_dngeo），
- dcn_levels=None 或 "all"：每步都用 DCN
- dcn_levels=[0]：仅步 0（如 P5→P4）用 DCN
- dcn_levels=[0, 1]：步 0、1 用 DCN，其余用 interpolate + add

geo_loss：用「top-down 之后、去噪之前」的 laterals 计算。
"""

from typing import List, Optional, Union

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
    """位置去噪模块：专注于去除 1×1 lateral 卷积引入的位置噪声."""

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

    def forward(self, lateral_features, backbone_features=None, levels_to_denoise=None):
        """levels_to_denoise: 仅对这些层做去噪，其余层原样返回；None 表示全部去噪。"""
        if levels_to_denoise is None:
            levels_to_denoise = list(range(len(lateral_features)))
        denoised_features = []
        for i, lateral_feat in enumerate(lateral_features):
            if i not in levels_to_denoise:
                denoised_features.append(lateral_feat)
                continue
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
    """在 FPN top-down 上采样融合处做特征对齐。"""

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
        self.offset_dim = 2 * k2 * self.deform_groups
        self.mask_dim = 1 * k2 * self.deform_groups
        self.offset_and_mask_dim = self.offset_dim + self.mask_dim

        self.offset_and_mask = nn.Conv2d(
            self.channels * 2,
            self.offset_and_mask_dim,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            bias=True,
        )

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
        normal_init(self.dcn, mean=0, std=0.01)
        if self.init_zero_offset:
            constant_init(self.offset_and_mask, 0)
        else:
            normal_init(self.offset_and_mask, mean=0, std=0.01)

    def forward(self, p_i, p_ip1_up):
        offset_and_mask = self.offset_and_mask(torch.cat([p_i, p_ip1_up], dim=1))
        offset = offset_and_mask[:, : self.offset_dim, :, :]
        mask = offset_and_mask[:, self.offset_dim :, :, :].sigmoid()
        aligned = self.dcn(p_ip1_up.contiguous(), offset, mask)
        return aligned


@NECKS.register_module()
class Reorder_DC_DNGeo_FPNAttention_LossPfive(FPN):
    """先 top-down(DCN) 再在部分层去噪的 DC_DNGeo FPN.

    流程：laterals → top-down(DCN) → 仅对 denoising_levels 去噪 → fpn_convs → outs
    可试 denoising_levels=[0, 1]。dcn_levels 控制哪些融合步用 DCN（类比 dc_last_dngeo）。
    geo_loss 用 top-down 之后、去噪之前的 laterals。
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
        use_denoising=False,
        denoising_weight=1.0,
        denoising_after_topdown: bool = True,
        denoising_levels: Optional[List[int]] = None,
        dcn_kernel_size: int = 3,
        dcn_deform_groups: int = 1,
        dcn_zero_init_offset: bool = True,
        dcn_levels: Optional[Union[List[int], str]] = None,
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
        self.denoising_after_topdown = denoising_after_topdown
        self.return_loss = return_loss
        self.contrastive_temp = contrastive_temp

        used_backbone_levels = self.backbone_end_level - self.start_level
        num_fusion_steps = used_backbone_levels - 1

        if dcn_levels is None or dcn_levels == "all":
            self.dcn_levels = list(range(num_fusion_steps))
        else:
            self.dcn_levels = list(dcn_levels)
            for k in self.dcn_levels:
                if k < 0 or k >= num_fusion_steps:
                    raise ValueError(
                        f"dcn_levels 中 {k} 超出范围 [0, {num_fusion_steps - 1}]"
                    )

        if denoising_levels is None:
            self.denoising_levels = [0, 1]
        else:
            self.denoising_levels = list(denoising_levels)
            for idx in self.denoising_levels:
                if idx < 0 or idx >= used_backbone_levels:
                    raise ValueError(
                        f"denoising_levels 中 {idx} 超出范围 [0, {used_backbone_levels - 1}]"
                    )

      

        if self.use_denoising:
            num_scales = len(self.lateral_convs)
            self.position_denoising_module = PositionDenoisingModule(
                in_channels=out_channels, num_scales=num_scales
            )

        self.align_modules = nn.ModuleList()
        for _ in range(used_backbone_levels-1):
            self.align_modules.append(
                _DCNAlign(
                    channels=out_channels,
                    kernel_size=dcn_kernel_size,
                    deform_groups=dcn_deform_groups,
                    init_zero_offset=dcn_zero_init_offset,
                )
            )

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
        """构建 lateral 特征。本模块流程为先 top-down 再部分层去噪，此处只返回原始 laterals。"""
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        if self.use_denoising and not self.denoising_after_topdown:
            denoised_laterals = self.position_denoising_module(laterals)
            laterals = [
                orig + self.denoising_weight * (denoised - orig)
                for orig, denoised in zip(laterals, denoised_laterals)
            ]
        return laterals

    def _build_topdown_and_outputs(self, laterals, inputs):
        """先完成 top-down(DCN)，得到融合后 laterals；若 denoising_after_topdown 且 use_denoising，
        仅对 denoising_levels 索引层做去噪，再 fpn_convs。返回 (outs, laterals_for_geo)。
        laterals_for_geo：用于 geo_loss，为 top-down 之后、去噪之前的 laterals（或未开启后去噪时为融合前 laterals）。
        """
        pre_topdown_laterals = laterals
        used_backbone_levels = len(laterals)

        for i in range(used_backbone_levels - 1, 0, -1):
            if "scale_factor" in self.upsample_cfg:
                up = F.interpolate(laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                up = F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)

            # 融合目标层为 i-1，对应步 k = i-1；仅 dcn_levels 指定步使用 DCN
            if (i - 1) in self.dcn_levels:
                aligned_up = self.align_modules[i - 1](laterals[i - 1], up)
                laterals[i - 1] = laterals[i - 1] + aligned_up
            else:
                laterals[i - 1] = laterals[i - 1] + up

        if self.denoising_after_topdown:
            laterals_for_geo = list(laterals)
            if self.use_denoising:
                denoised_laterals = self.position_denoising_module(
                    laterals, levels_to_denoise=self.denoising_levels
                )
                for i in self.denoising_levels:
                    if i < len(laterals):
                        orig = laterals[i]
                        laterals[i] = orig + self.denoising_weight * (
                            denoised_laterals[i] - orig
                        )
        else:
            laterals_for_geo = pre_topdown_laterals

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

        return tuple(outs), laterals_for_geo

    @auto_fp16()
    def forward(self, inputs, backbone_features=None):
        assert len(inputs) == len(self.in_channels)

        laterals = self._build_laterals_with_denoising(inputs)
        fpn_outs, laterals_for_geo = self._build_topdown_and_outputs(laterals, inputs)

        if self.training and self.return_loss and backbone_features is not None:
            losses = self._compute_geo_loss(laterals_for_geo, backbone_features)
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

        if len(x_lateral) >= 3:
            levels = list(range(len(x_lateral)))[2:]
        else:
            levels = [len(x_lateral) - 1] if len(x_lateral) > 0 else []

        if len(levels) == 0:
            device = x_lateral[0].device
            return {"geo_loss": torch.tensor(0.0, device=device)}

        batch_size = x_lateral[0].shape[0]
        device = x_lateral[0].device

        lateral_loc, backbone_loc = [], []
        for lvl in levels:
            lateral_loc.append(self.lateral_loc_encoder(x_lateral[lvl]))
            backbone_aligned = self.channel_transfer[lvl](x_b[lvl])
            backbone_loc.append(self.backbone_loc_encoder(backbone_aligned))

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

        neg_mask = (~scale_mask) & (~batch_mask)

        if neg_mask.sum() == 0:
            neg_mask = ~pos_mask
            neg_mask = neg_mask.fill_diagonal_(False)

        exp_sim_lateral_backbone = torch.exp(sim_lateral_backbone)
        pos_sim = exp_sim_lateral_backbone * pos_mask
        numerators = pos_sim.sum(dim=1)

        exp_sim_lateral_lateral = torch.exp(sim_lateral_lateral)
        neg_sim = (exp_sim_lateral_lateral + exp_sim_lateral_backbone) * neg_mask
        denominators = neg_sim.sum(dim=1) + numerators

        if denominators.sum() == 0 or torch.isnan(denominators).any():
            denominators = numerators + exp_sim_lateral_lateral.sum(dim=1)

        geo_loss = -torch.log(numerators / (denominators + 1e-8) + 1e-8).mean()
        return {"geo_loss": geo_loss}


Reorder_DC_DNGeo_FPN_Attention_LossPfive = Reorder_DC_DNGeo_FPNAttention_LossPfive


# ---------------------------------------------------------------------------
# 3.3 配置示例
# ---------------------------------------------------------------------------
# 开启「先 top-down 再在部分层去噪」，仅对索引 0、1 层做融合后去噪：
#
# neck=dict(
#     type='Reorder_DC_DNGeo_FPNAttention_LossPfive',
#     in_channels=[256, 512, 1024, 2048],
#     out_channels=256,
#     num_outs=5,
#     use_denoising=True,
#     denoising_weight=1.0,
#     denoising_after_topdown=True,
#     denoising_levels=[0, 1],
#     dcn_levels=[0, 1],  # 仅步 0、1 用 DCN
#     return_loss=True,
# )
#
# 与原始 dc_dngeo 行为一致（先去噪再 top-down）：
# denoising_after_topdown=False（默认）
# denoising_after_topdown=False（默认）
