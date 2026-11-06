"""


在 dc_dngeo_fpn_attention_losspfive 基础上的变体：
- 对比学习（geo_loss）使用**上采样并 DCN 对齐后**的特征，而非 1×1 卷积后的 lateral 特征。
- 即：FeatureEncoder 的输入为 top-down 路径中“从上一尺度上采样并对齐后”的特征，与 backbone 特征做对比。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, constant_init, normal_init
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d
from mmcv.runner import auto_fp16

from ..builder import NECKS
from .dc_dngeo_fpn_attention_losspfive import (
    DC_DNGeo_FPNAttention_LossPfive,
    FeatureEncoder,
    PositionDenoisingModule,
    _DCNAlign,
)


@NECKS.register_module()
class DC_DNGeo_FPNAttention_LossPfive_LossP(DC_DNGeo_FPNAttention_LossPfive):
    """对比学习使用上采样后特征的 DNGeo FPN 变体.

    与 DC_DNGeo_FPNAttention_LossPfive 的唯一区别：
    - geo_loss 中用于对比的 FPN 侧特征来自「上采样 + DCN 对齐」后的特征，
      而非 1×1 lateral 卷积后的特征。
    """

    def _build_topdown_and_outputs(self, laterals, inputs):
        """构建 top-down 路径（带 DCN 对齐），收集每层的上采样对齐特征，并生成输出."""
        used_backbone_levels = len(laterals)
        # 保存每层「从上一尺度上采样并 DCN 对齐后」的特征，用于对比学习
        # upsampled_features[i] = 加到 level i 上的 aligned(up(laterals[i+1]))，长度 used_backbone_levels - 1
        upsampled_features = [None] * (used_backbone_levels - 1)

        for i in range(used_backbone_levels - 1, 0, -1):
            if "scale_factor" in self.upsample_cfg:
                up = F.interpolate(laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                up = F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)

            aligned_up = self.align_modules[i - 1](laterals[i - 1], up)
            laterals[i - 1] = laterals[i - 1] + aligned_up
            upsampled_features[i - 1] = aligned_up

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

        return tuple(outs), upsampled_features

    @auto_fp16()
    def forward(self, inputs, backbone_features=None):
        assert len(inputs) == len(self.in_channels)

        laterals = self._build_laterals_with_denoising(inputs)
        fpn_outs, upsampled_features = self._build_topdown_and_outputs(laterals, inputs)

        if self.training and self.return_loss and backbone_features is not None:
            losses = self._compute_geo_loss_upsampled(upsampled_features, backbone_features)
            return fpn_outs, losses

        return fpn_outs

    def _compute_geo_loss_upsampled(self, upsampled_features, backbone_features):
        """使用上采样后的特征计算几何对比学习损失（levels 仍为 [2:] 等）。"""
        x_b = backbone_features[self.start_level : self.backbone_end_level]
        # 仅在有上采样特征的尺度上计算：upsampled_features[i] 对应 level i（来自 level i+1 的上采样）
        n_upsampled = len(upsampled_features)
        if n_upsampled == 0:
            device = (
                backbone_features[0].device
                if backbone_features
                else torch.device("cpu")
            )
            return {"geo_loss": torch.tensor(0.0, device=device)}

        n = min(len(x_b), n_upsampled)
        x_b = x_b[:n]
        x_upsampled = list(upsampled_features[:n])

        # 参与 geo_loss 的尺度索引（与 losspfive 一致：levels=[2:]）
        if len(x_upsampled) >= 3:
            levels = list(range(len(x_upsampled)))[2:]
        else:
            levels = [len(x_upsampled) - 1] if len(x_upsampled) > 0 else []

        if len(levels) == 0:
            device = x_upsampled[0].device
            return {"geo_loss": torch.tensor(0.0, device=device)}

        batch_size = x_upsampled[0].shape[0]
        device = x_upsampled[0].device

        upsampled_loc, backbone_loc = [], []
        for lvl in levels:
            upsampled_loc.append(self.lateral_loc_encoder(x_upsampled[lvl]))
            backbone_aligned = self.channel_transfer[lvl](x_b[lvl])
            backbone_loc.append(self.backbone_loc_encoder(backbone_aligned))

        loc_upsampled_all = torch.stack(upsampled_loc, dim=0).view(-1, upsampled_loc[0].shape[-1])
        loc_backbone_all = torch.stack(backbone_loc, dim=0).view(-1, backbone_loc[0].shape[-1])

        loc_upsampled_norm = F.normalize(loc_upsampled_all, p=2, dim=1)
        loc_backbone_norm = F.normalize(loc_backbone_all, p=2, dim=1)

        sim_upsampled_upsampled = (
            torch.matmul(loc_upsampled_norm, loc_upsampled_norm.t()) / self.contrastive_temp
        )
        sim_upsampled_backbone = (
            torch.matmul(loc_upsampled_norm, loc_backbone_norm.t()) / self.contrastive_temp
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

        exp_sim_upsampled_backbone = torch.exp(sim_upsampled_backbone)
        pos_sim = exp_sim_upsampled_backbone * pos_mask
        numerators = pos_sim.sum(dim=1)

        exp_sim_upsampled_upsampled = torch.exp(sim_upsampled_upsampled)
        neg_sim = (exp_sim_upsampled_upsampled + exp_sim_upsampled_backbone) * neg_mask
        denominators = neg_sim.sum(dim=1) + numerators

        if denominators.sum() == 0 or torch.isnan(denominators).any():
            denominators = numerators + exp_sim_upsampled_upsampled.sum(dim=1)

        geo_loss = -torch.log(numerators / (denominators + 1e-8) + 1e-8).mean()
        return {"geo_loss": geo_loss}


# 兼容别名
DC_DNGeo_FPN_Attention_LossPfive_LossP = DC_DNGeo_FPNAttention_LossPfive_LossP
