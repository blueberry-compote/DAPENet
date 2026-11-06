"""
dnsem_fpn_pure.py

纯净版 DNSem FPN：
- 不包含任何注意力 / 去噪模块
- 仅保留基于对比学习的 sem_loss（用于约束相邻尺度 FPN 输出的语义一致性）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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


@NECKS.register_module()
class DNSem_FPNPure(FPN):
    """无注意力/无去噪，仅返回 sem_loss 的 DNSem FPN 版本.

    训练时（self.training=True 且 return_loss=True）返回：
    - fpn_outs: Tuple[Tensor]
    - losses: {"sem_loss": Tensor}
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

        self.return_loss = return_loss
        self.contrastive_temp = contrastive_temp

        if return_loss:
            # 用于编码融合后的 FPN 特征（语义特征编码器）
            self.fpn_sem_encoder = FeatureEncoder(out_channels)

    @auto_fp16()
    def forward(self, inputs, backbone_features=None):
        # 直接走标准 FPN 流程（lateral + topdown + 3x3 fpn_convs）
        fpn_outs = super().forward(inputs)

        if self.training and self.return_loss:
            losses = self._compute_sem_loss(fpn_outs)
            return fpn_outs, losses

        return fpn_outs

    def _compute_sem_loss(self, fpn_features):
        """语义对齐损失（sem_loss）：相同 batch、相邻尺度的 FPN 特征应该相似."""

        x_fpn = list(fpn_features)[:4]  # 只处理前4层 FPN 特征（与 dnsem_fpn.py 对齐）
        scale_size = len(x_fpn)

        if scale_size <= 1:
            device = (
                fpn_features[0].device
                if len(fpn_features) > 0
                else torch.device("cpu")
            )
            return {"sem_loss": torch.tensor(0.0, device=device)}

        batch_size = x_fpn[0].shape[0]
        device = x_fpn[0].device

        # 编码每个尺度的 FPN 输出为向量表征
        semantic_fpn = [self.fpn_sem_encoder(x_fpn[i]) for i in range(scale_size)]  # List[[B,D]]

        # anchors: 前 scale_size-1 个尺度（与后一个相邻尺度构成正样本）
        sem_fpn_anchors = torch.cat(semantic_fpn[:-1], dim=0)  # [M, D], M=(S-1)*B
        sem_fpn_anchors_norm = F.normalize(sem_fpn_anchors, p=2, dim=1)

        # keys: 所有尺度
        all_fpn_keys = torch.cat(semantic_fpn, dim=0)  # [N, D], N=S*B
        all_keys_norm = F.normalize(all_fpn_keys, p=2, dim=1)

        # 相似度矩阵 [M, N]
        sim_matrix = torch.matmul(sem_fpn_anchors_norm, all_keys_norm.t()) / self.contrastive_temp

        # 正样本掩码：同 batch 且 key_scale = anchor_scale + 1
        anchor_scale_indices = torch.arange(scale_size - 1, device=device).repeat_interleave(
            batch_size
        )  # [M]
        anchor_batch_indices = torch.arange(batch_size, device=device).repeat(scale_size - 1)  # [M]

        key_scale_indices = torch.arange(scale_size, device=device).repeat_interleave(batch_size)  # [N]
        key_batch_indices = torch.arange(batch_size, device=device).repeat(scale_size)  # [N]

        is_same_batch = anchor_batch_indices.unsqueeze(1) == key_batch_indices.unsqueeze(0)  # [M,N]
        is_adjacent_scale = (anchor_scale_indices.unsqueeze(1) + 1) == key_scale_indices.unsqueeze(0)  # [M,N]
        pos_mask_sem = is_same_batch & is_adjacent_scale

        exp_sim = torch.exp(sim_matrix)
        numerators = (exp_sim * pos_mask_sem).sum(dim=1)  # [M]
        denominators = exp_sim.sum(dim=1)  # [M]

        sem_loss = -torch.log(numerators / (denominators + 1e-8) + 1e-8).mean()
        return {"sem_loss": sem_loss}

