"""Plain FPN + old FE (CBR + spatial attn), no DDA / no memory.

Keeps standard FPN top-down fusion, then applies PositionDenoising-style FE
on selected levels: CBR -> Conv7x7 sigmoid spatial gate -> residual blend.
Channel attention is off by default for this ablation suite.
"""
from typing import List, Optional

import torch.nn.functional as F
from mmcv.runner import auto_fp16

from ..builder import NECKS
from .fpn import FPN
from .reorder_dc_dngeo_shared_memory_attn_ablation import (
    PositionDenoisingModuleAblation,
)


@NECKS.register_module()
class PlainFPNOldFE(FPN):
    """Plain FPN top-down + old FE (CBR + spatial attention)."""

    def __init__(
        self,
        *args,
        denoising_levels: Optional[List[int]] = None,
        denoising_weight: float = 1.0,
        use_spatial_attn: bool = True,
        use_channel_attn: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.denoising_levels = (
            [0, 1, 2] if denoising_levels is None else list(denoising_levels)
        )
        self.denoising_weight = denoising_weight
        self.use_spatial_attn = use_spatial_attn
        self.use_channel_attn = use_channel_attn
        self.position_denoising_module = PositionDenoisingModuleAblation(
            in_channels=self.out_channels,
            num_scales=len(self.lateral_convs),
            use_spatial_attn=use_spatial_attn,
            use_channel_attn=use_channel_attn,
        )

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        laterals = [
            lateral_conv(inputs[level + self.start_level])
            for level, lateral_conv in enumerate(self.lateral_convs)
        ]
        used_backbone_levels = len(laterals)

        # Standard FPN top-down addition; no dynamic deformable alignment.
        for level in range(used_backbone_levels - 1, 0, -1):
            target_shape = laterals[level - 1].shape[2:]
            upsampled = F.interpolate(
                laterals[level],
                size=target_shape,
                **self.upsample_cfg,
            )
            laterals[level - 1] = laterals[level - 1] + upsampled

        # Old FE: CBR + spatial attention + residual blend on P3/P4/P5.
        denoised_laterals = self.position_denoising_module(
            laterals, levels_to_denoise=self.denoising_levels)
        for i in self.denoising_levels:
            if i < len(laterals):
                original = laterals[i]
                laterals[i] = original + self.denoising_weight * (
                    denoised_laterals[i] - original)

        outs = [
            self.fpn_convs[level](laterals[level])
            for level in range(used_backbone_levels)
        ]

        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for _ in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError

                outs.append(
                    self.fpn_convs[used_backbone_levels](extra_source))
                for level in range(
                    used_backbone_levels + 1, self.num_outs
                ):
                    if self.relu_before_extra_convs:
                        outs.append(
                            self.fpn_convs[level](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[level](outs[-1]))

        return tuple(outs)
