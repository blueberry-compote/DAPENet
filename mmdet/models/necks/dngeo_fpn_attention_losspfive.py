"""
dngeo_fpn_attention_losspfive.py

基于 `DNGeo_FPNPure_pfive` 的版本：
- 保留 pure_pfive 的 geo_loss 计算逻辑（仅在部分尺度上做对比学习，默认 levels=[2:]）
- 在 1×1 lateral 之后、top-down 之前，引入 `DNGeo_FPN` 中的 `PositionDenoisingModule`
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
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


@NECKS.register_module()
class DNGeo_FPNAttention_LossPfive(FPN):
    """在 pure_pfive 基础上加入 PositionDenoisingModule 的 DNGeo FPN 版本."""

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
        use_denoising=True,
        denoising_weight=1.0,
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

        if self.use_denoising:
            # 对应的 lateral 数量等于 backbone 使用的层数（不是 num_outs）
            num_scales = len(self.lateral_convs)
            self.position_denoising_module = PositionDenoisingModule(
                in_channels=out_channels, num_scales=num_scales
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
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            if "scale_factor" in self.upsample_cfg:
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg
                )
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg
                )

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

        laterals = self._build_laterals_with_denoising(inputs)
        fpn_outs = self._build_topdown_and_outputs(laterals, inputs)

        if self.training and self.return_loss and backbone_features is not None:
            losses = self._compute_geo_loss(laterals, backbone_features)
            return fpn_outs, losses

        return fpn_outs

    def _compute_geo_loss(self, lateral_features, backbone_features):
        # 复用 `DNGeo_FPNPure_pfive` 的 geo_loss 逻辑
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
        levels = list(range(len(x_lateral)))[2:]
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

        # 与 pure_pfive 一致：仅把“尺度不同 且 batch 不同”的样本当作负样本
        neg_mask = (~scale_mask) & (~batch_mask)

        exp_sim_lateral_backbone = torch.exp(sim_lateral_backbone)
        pos_sim = exp_sim_lateral_backbone * pos_mask
        numerators = pos_sim.sum(dim=1)

        exp_sim_lateral_lateral = torch.exp(sim_lateral_lateral)
        neg_sim = (exp_sim_lateral_lateral + exp_sim_lateral_backbone) * neg_mask
        denominators = neg_sim.sum(dim=1) + numerators

        geo_loss = -torch.log(numerators / (denominators + 1e-8) + 1e-8).mean()
        return {"geo_loss": geo_loss}


# 兼容可能的 type 命名习惯（可选别名）
DNGeo_FPN_Attention_LossPfive = DNGeo_FPNAttention_LossPfive
