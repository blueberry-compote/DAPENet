"""
dc_last_dngeo_lossP_fpn_attention_losspfive.py

在 dc_last_dngeo_fpn_attention_losspfive 与 dc_dngeo_lossP_fpn_attention_losspfive 基础上的结合：

1. 来自 dc_last_dngeo_fpn_attention_losspfive：
   - dcn_levels：可选择在哪些 top-down 融合步使用可变形卷积（DCN），其余步用普通 interpolate + add。

2. 来自 dc_dngeo_lossP_fpn_attention_losspfive：
   - 对比学习（geo_loss）使用「上采样并（若该步使用 DCN 则）DCN 对齐后」的特征，
     与 backbone 特征做对比，而非 1×1 lateral 后的特征。
"""

import torch
import torch.nn.functional as F
from mmcv.runner import auto_fp16

from ..builder import NECKS
from .dc_last_dngeo_fpn_attention_losspfive import (
    DC_Last_DNGeo_FPNAttention_LossPfive,
)


@NECKS.register_module()
class DC_Last_DNGeo_FPNAttention_LossPfive_LossP(DC_Last_DNGeo_FPNAttention_LossPfive):
    """支持 dcn_levels + 对比学习使用上采样后特征的 DNGeo FPN 变体.

    - dcn_levels：仅在其指定的融合步使用 DCN，其余步普通上采样相加（与 DC_Last_DNGeo 一致）。
    - geo_loss：用于对比的 FPN 侧特征为「该步实际加上的特征」——若该步用 DCN 则为对齐后的特征，
      否则为普通上采样特征，与 backbone 做对比学习。
    """

    def _build_topdown_and_outputs(self, laterals, inputs):
        """构建 top-down 路径；仅在 dcn_levels 指定步使用 DCN，并收集每步「加上的」特征供对比学习."""
        used_backbone_levels = len(laterals)
        # upsampled_features[k] = 在 level k 上加上的特征（来自 level k+1 的上采样，可能经 DCN 对齐）
        upsampled_features = [None] * (used_backbone_levels - 1)

        for i in range(used_backbone_levels - 1, 0, -1):
            if "scale_factor" in self.upsample_cfg:
                up = F.interpolate(laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                up = F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)

            step_k = i - 1
            if step_k in self.dcn_levels:
                aligned_up = self.align_modules[step_k](laterals[i - 1], up)
                laterals[i - 1] = laterals[i - 1] + aligned_up
                upsampled_features[step_k] = aligned_up
            else:
                laterals[i - 1] = laterals[i - 1] + up
                upsampled_features[step_k] = up

        outs = [self.fpn_convs[j](laterals[j]) for j in range(used_backbone_levels)]

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
                for j in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[j](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[j](outs[-1]))

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
        """使用上采样后（及可能 DCN 对齐后）的特征计算几何对比学习损失（levels 仍为 [2:] 等）。"""
        x_b = backbone_features[self.start_level : self.backbone_end_level]
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
DC_Last_DNGeo_FPN_Attention_LossPfive_LossP = DC_Last_DNGeo_FPNAttention_LossPfive_LossP
