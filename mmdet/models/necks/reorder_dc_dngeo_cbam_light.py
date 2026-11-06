"""Reorder DC-DNGeo Lite with standard CBAM spatial attention in PositionDenoisingModule."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16
from typing import List, Optional, Union

from ..builder import NECKS
from .fpn import FPN
from .reorder_dc_dngeo_light import _DCNAlign


class CBAMSpatialAttention(nn.Module):
    """标准 CBAM 空间注意力：通道维 avg/max 拼接后经 7×7 conv + Sigmoid。"""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return torch.sigmoid(out)


class PositionDenoisingModule(nn.Module):
    """位置去噪模块，空间注意力采用标准 CBAM 实现。"""

    def __init__(self, in_channels=256, num_scales=4):
        super().__init__()
        self.num_scales = num_scales
        self.denoising_convs = nn.ModuleList([
            ConvModule(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=dict(type="BN"),
                act_cfg=dict(type="ReLU", inplace=True),
            ) for _ in range(num_scales)
        ])

        self.spatial_attentions = nn.ModuleList([
            CBAMSpatialAttention(kernel_size=7) for _ in range(num_scales)
        ])

        self.channel_attentions = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, in_channels // 16, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 16, in_channels, 1),
                nn.Sigmoid(),
            ) for _ in range(num_scales)
        ])

    def forward(self, lateral_features, levels_to_denoise=None):
        if levels_to_denoise is None:
            levels_to_denoise = list(range(len(lateral_features)))
        denoised_features = []
        for i, lateral_feat in enumerate(lateral_features):
            if i not in levels_to_denoise:
                denoised_features.append(lateral_feat)
                continue

            denoised = self.denoising_convs[i](lateral_feat)
            denoised = denoised * self.spatial_attentions[i](denoised)
            denoised = denoised * self.channel_attentions[i](denoised)
            denoised_features.append(denoised)

        return denoised_features


@NECKS.register_module()
class Reorder_DC_DNGeo_CBAM_Lite(FPN):
    """流程：laterals -> top-down(DCN Align) -> 部分层去噪(CBAM 空间注意力) -> fpn_convs -> outs"""

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
        denoising_after_topdown: bool = True,
        denoising_levels: Optional[List[int]] = None,
        dcn_kernel_size: int = 3,
        dcn_deform_groups: int = 1,
        dcn_zero_init_offset: bool = True,
        dcn_levels: Optional[Union[List[int], str]] = None,
        dcn_mask_tau: float = 0.03,
        dcn_gate_mode: str = "soft",
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
        self.denoising_after_topdown = denoising_after_topdown

        used_backbone_levels = self.backbone_end_level - self.start_level
        num_fusion_steps = used_backbone_levels - 1

        if dcn_levels is None or dcn_levels == "all":
            self.dcn_levels = list(range(num_fusion_steps))
        else:
            self.dcn_levels = list(dcn_levels)

        if denoising_levels is None:
            self.denoising_levels = [0, 1]
        else:
            self.denoising_levels = list(denoising_levels)

        if self.use_denoising:
            self.position_denoising_module = PositionDenoisingModule(
                in_channels=out_channels, num_scales=len(self.lateral_convs)
            )

        self.align_modules = nn.ModuleList([
            _DCNAlign(
                channels=out_channels,
                kernel_size=dcn_kernel_size,
                deform_groups=dcn_deform_groups,
                init_zero_offset=dcn_zero_init_offset,
                mask_tau=dcn_mask_tau,
                gate_mode=dcn_gate_mode,
            ) for _ in range(num_fusion_steps)
        ])

    def get_keep_ratio(self):
        """返回当前 batch 的平均激活比例（用于日志观测，不参与反传）。"""
        keep_ratio = None
        count = 0
        for m in self.align_modules:
            if hasattr(m, "last_keep_ratio"):
                cur = m.last_keep_ratio
                keep_ratio = cur if keep_ratio is None else (keep_ratio + cur)
                count += 1
        if count > 0:
            return keep_ratio / count
        return self.align_modules[0].offset_and_mask.weight.new_tensor(0.0)

    def _build_laterals_with_denoising(self, inputs):
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        if self.use_denoising and not self.denoising_after_topdown:
            laterals = self.position_denoising_module(laterals)
        return laterals

    def _build_topdown_and_outputs(self, laterals, inputs):
        pre_topdown_laterals = [l.clone() for l in laterals]
        used_backbone_levels = len(laterals)

        for i in range(used_backbone_levels - 1, 0, -1):
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
                    laterals[i] = denoised_laterals[i]

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
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        laterals = self._build_laterals_with_denoising(inputs)
        fpn_outs, _ = self._build_topdown_and_outputs(laterals, inputs)
        return fpn_outs
