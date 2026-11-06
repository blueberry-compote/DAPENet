"""
reorder_dc_dngeo_fpn_attention_losspfive_weights.py

在原方案 3（先 top-down 再在部分层去噪）的基础上，引入：
- CBAM 风格的通道注意力（AvgPool + MaxPool）
- 特征自适应的去噪门控权重（W，而非标量 weight）
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


class CBAMChannelAttention(nn.Module):
    """CBAM 风格的通道注意力：AvgPool + MaxPool + 共享 MLP."""

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        mid_channels = max(in_channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(F.adaptive_avg_pool2d(x, 1))
        max_out = self.mlp(F.adaptive_max_pool2d(x, 1))
        out = avg_out + max_out
        return torch.sigmoid(out)


class PositionDenoisingModule(nn.Module):
    """位置去噪模块（自适应权重 + CBAM 风格注意力）.

    - 先用 3×3 Conv 做局部去噪
    - 再用空间注意力（7×7）和 CBAM 通道注意力自适应增强
    - 最后用特征自适应门控 W 做残差混合：
        out = W ⊙ denoised + (1 - W) ⊙ lateral
    """

    def __init__(self, in_channels: int = 256, num_scales: int = 4):
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

        # 空间注意力保持与原实现一致：7×7 conv + Sigmoid
        self.spatial_attentions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),
                    nn.Sigmoid(),
                )
                for _ in range(num_scales)
            ]
        )

        # 使用 CBAM 风格的通道注意力（AvgPool + MaxPool + 共享 MLP）
        self.channel_attentions = nn.ModuleList(
            [CBAMChannelAttention(in_channels) for _ in range(num_scales)]
        )

        # 特征自适应的去噪门控：通道门控 W_c ∈ [0,1]^{B×C×1×1}
        self.denoising_gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(in_channels, in_channels, kernel_size=1),
                    nn.Sigmoid(),
                )
                for _ in range(num_scales)
            ]
        )

    def forward(
        self,
        lateral_features: List[torch.Tensor],
        backbone_features: Optional[List[torch.Tensor]] = None,
        levels_to_denoise: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        """levels_to_denoise: 仅对这些层做去噪，其余层原样返回；None 表示全部去噪。"""
        if levels_to_denoise is None:
            levels_to_denoise = list(range(len(lateral_features)))

        denoised_features: List[torch.Tensor] = []
        for i, lateral_feat in enumerate(lateral_features):
            if i not in levels_to_denoise:
                denoised_features.append(lateral_feat)
                continue

            # 3×3 conv 去噪
            denoised = self.denoising_convs[i](lateral_feat)

            # 空间注意力
            spatial_att = self.spatial_attentions[i](denoised)
            denoised = denoised * spatial_att

            # CBAM 通道注意力
            channel_att = self.channel_attentions[i](denoised)
            denoised = denoised * channel_att

            # 特征自适应门控（通道门控）
            gate = self.denoising_gates[i](denoised)
            denoised_feat = gate * denoised + (1.0 - gate) * lateral_feat
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
        assert kernel_size in (1, 3), "当前实现建议 kernel_size=1 或 3"
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

    def _init_weights(self) -> None:
        normal_init(self.dcn, mean=0, std=0.01)
        if self.init_zero_offset:
            constant_init(self.offset_and_mask, 0)
        else:
            normal_init(self.offset_and_mask, mean=0, std=0.01)

    def forward(self, p_i: torch.Tensor, p_ip1_up: torch.Tensor) -> torch.Tensor:
        offset_and_mask = self.offset_and_mask(torch.cat([p_i, p_ip1_up], dim=1))
        offset = offset_and_mask[:, : self.offset_dim, :, :]
        mask = offset_and_mask[:, self.offset_dim :, :, :].sigmoid()
        aligned = self.dcn(p_ip1_up.contiguous(), offset, mask)
        return aligned


@NECKS.register_module()
class Reorder_DC_DNGeo_FPN_Attention_Weights(FPN):
    """先 top-down(DCN) 再在部分层去噪的 DC_DNGeo FPN（自适应去噪权重 + CBAM 注意力）.

    流程：laterals → top-down(DCN) → 仅对 denoising_levels 去噪 → fpn_convs → outs
    可试 denoising_levels=[0, 1]。dcn_levels 控制哪些融合步用 DCN（类比 dc_last_dngeo）。
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
        use_denoising=True,
        denoising_weight=1.0,
        denoising_after_topdown: bool = True,
        denoising_levels: Optional[List[int]] = None,
        dcn_kernel_size: int = 3,
        dcn_deform_groups: int = 1,
        dcn_zero_init_offset: bool = True,
        dcn_levels: Optional[Union[List[int], str]] = None,
        return_loss=True,
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
        # 已移除 geo_loss；保留入参仅为兼容旧配置文件
        _ = return_loss

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
        for _ in range(used_backbone_levels - 1):
            self.align_modules.append(
                _DCNAlign(
                    channels=out_channels,
                    kernel_size=dcn_kernel_size,
                    deform_groups=dcn_deform_groups,
                    init_zero_offset=dcn_zero_init_offset,
                )
            )

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
        仅对 denoising_levels 索引层做去噪，再 fpn_convs。返回 outs。
        """

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

        return fpn_outs




