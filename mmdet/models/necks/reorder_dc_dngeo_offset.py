"""
reorder_dc_dngeo_offset.py

基于 reorder_dc_dngeo.py，将 DCN 对齐模块改为带 offset 瓶颈的版本，以降低 offset 分支参数量。

流程与 reorder_dc_dngeo 一致：laterals → top-down(DCN with offset bottleneck) → 去噪 → fpn_convs → outs。
对齐模块使用 2C → mid_channels → offset_and_mask_dim 的瓶颈结构，mid_channels = C // offset_bottleneck_ratio（默认 4）。
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


class DCNAlignWithOffsetBottleneck(nn.Module):
    """带 offset 瓶颈的 DCN 对齐模块，接口与原 `_DCNAlign` 保持兼容。

    相比直接用 `Conv2d(2C -> offset_and_mask_dim)`，这里先用较小的中间通道
    做一个瓶颈结构，以减少 offset 分支参数量，同时尽量保持对齐效果。
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        deform_groups: int = 1,
        init_zero_offset: bool = True,
        offset_bottleneck_ratio: int = 4,
        offset_bottleneck_channels: Optional[int] = None,
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

        # 如果显式指定了瓶颈通道数，则优先使用；否则按 ratio 计算一个安全下界
        if offset_bottleneck_channels is None:
            mid_channels = max(self.channels // offset_bottleneck_ratio, 16)
        else:
            mid_channels = int(offset_bottleneck_channels)
        self.mid_channels = mid_channels

        # offset/mask 瓶颈结构：2C -> mid -> offset_and_mask_dim
        self.offset_and_mask = nn.Sequential(
            nn.Conv2d(
                self.channels * 2,
                self.mid_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.mid_channels,
                self.offset_and_mask_dim,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                bias=True,
            ),
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
        last_offset_conv = (
            self.offset_and_mask[2]
            if isinstance(self.offset_and_mask, nn.Sequential)
            and len(self.offset_and_mask) >= 3
            else None
        )
        if last_offset_conv is not None:
            if self.init_zero_offset:
                constant_init(last_offset_conv, 0)
            else:
                normal_init(last_offset_conv, mean=0, std=0.01)

    def forward(self, p_i, p_ip1_up):
        """p_i: 当前层特征；p_ip1_up: 上一层上采样后的特征."""
        offset_and_mask = self.offset_and_mask(torch.cat([p_i, p_ip1_up], dim=1))
        offset = offset_and_mask[:, : self.offset_dim, :, :]
        mask = offset_and_mask[:, self.offset_dim :, :, :].sigmoid()
        aligned = self.dcn(p_ip1_up.contiguous(), offset, mask)
        return aligned


@NECKS.register_module()
class Reorder_DC_DNGeo_Offset(FPN):
    """先 top-down(DCN with offset bottleneck) 再在部分层去噪的 DC_DNGeo FPN.

    与 Reorder_DC_DNGeo 行为一致，仅将对齐模块换为带 offset 瓶颈的 DCN，用于减少参数量。
    流程：laterals → top-down(DCN) → 仅对 denoising_levels 去噪 → fpn_convs → outs。
    不包含几何对比学习损失，仅返回 FPN 特征。
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
        dcn_offset_bottleneck_ratio: int = 4,
        dcn_offset_bottleneck_channels: Optional[int] = None,
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
                in_channels=out_channels,
                num_scales=num_scales,
            )

        self.align_modules = nn.ModuleList()
        for _ in range(used_backbone_levels - 1):
            self.align_modules.append(
                DCNAlignWithOffsetBottleneck(
                    channels=out_channels,
                    kernel_size=dcn_kernel_size,
                    deform_groups=dcn_deform_groups,
                    init_zero_offset=dcn_zero_init_offset,
                    offset_bottleneck_ratio=dcn_offset_bottleneck_ratio,
                    offset_bottleneck_channels=dcn_offset_bottleneck_channels,
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
        仅对 denoising_levels 索引层做去噪，再 fpn_convs。返回 (outs, pre_topdown_laterals)。
        """
        pre_topdown_laterals = laterals
        used_backbone_levels = len(laterals)

        for i in range(used_backbone_levels - 1, 0, -1):
            if "scale_factor" in self.upsample_cfg:
                up = F.interpolate(laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                up = F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)

            if (i - 1) in self.dcn_levels:
                aligned_up = self.align_modules[i - 1](laterals[i - 1], up)
                laterals[i - 1] = laterals[i - 1] + aligned_up
            else:
                laterals[i - 1] = laterals[i - 1] + up

        if self.denoising_after_topdown and self.use_denoising:
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

        return tuple(outs), pre_topdown_laterals

    @auto_fp16()
    def forward(self, inputs, backbone_features=None):
        assert len(inputs) == len(self.in_channels)

        laterals = self._build_laterals_with_denoising(inputs)
        fpn_outs, _ = self._build_topdown_and_outputs(laterals, inputs)

        return fpn_outs
