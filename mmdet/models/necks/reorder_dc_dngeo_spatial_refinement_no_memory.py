"""DDA + RC-SA operators without any detail-memory retrieval.

Keeps dynamic deformable alignment, then applies the same post-fusion
refinement stack as AblationSpatialRefinement (CBR + spatial gate without
V / s).  There is no memory bank and no prototype injection.
"""
from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16

from ..builder import NECKS
from .plain_fpn_spatial_refinement import AblationSpatialRefinement
from .reorder_dc_dngeo_light import Reorder_DC_DNGeo_Lite


@NECKS.register_module()
class Reorder_DC_DNGeo_SpatialRefinement_NoMemory(Reorder_DC_DNGeo_Lite):
    """DDA neck with RC-SA-aligned spatial refinement and no retrieval."""

    def __init__(
        self,
        *args,
        refinement_levels: Optional[List[int]] = None,
        gate_bias_init: float = 0.0,
        **kwargs,
    ):
        # Drop the original FE path; RC-SA operators replace it.
        kwargs['use_denoising'] = False
        super().__init__(*args, **kwargs)

        self.refinement_levels = (
            [0, 1, 2]
            if refinement_levels is None
            else list(refinement_levels)
        )
        self.refinement_modules = nn.ModuleDict({
            str(level): AblationSpatialRefinement(
                channels=self.out_channels,
                gate_bias_init=gate_bias_init,
            )
            for level in self.refinement_levels
        })

    def _build_topdown_and_outputs(self, laterals, inputs):
        pre_topdown_laterals = [l.clone() for l in laterals]
        used_backbone_levels = len(laterals)

        # 1. Top-down fusion with DDA on configured levels.
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            up = F.interpolate(
                laterals[i], size=prev_shape, **self.upsample_cfg)

            if (i - 1) in self.dcn_levels:
                aligned_up = self.align_modules[i - 1](
                    laterals[i - 1], up)
                laterals[i - 1] = laterals[i - 1] + aligned_up
            else:
                laterals[i - 1] = laterals[i - 1] + up

        # 2. RC-SA operators without retrieval inputs (no V / s).
        for level in self.refinement_levels:
            if level < 0 or level >= len(laterals):
                raise IndexError(
                    f'refinement level {level} is out of range for '
                    f'{len(laterals)} laterals')
            laterals[level] = self.refinement_modules[str(level)](
                laterals[level])

        # 3. Output convolutions for P3-P5.
        outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels)
        ]

        # 4. Extra high-level outputs P6/P7...
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
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(
                            self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        return tuple(outs), pre_topdown_laterals

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        laterals = self._build_laterals_with_denoising(inputs)
        fpn_outs, _ = self._build_topdown_and_outputs(laterals, inputs)
        return fpn_outs
