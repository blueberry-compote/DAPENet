"""Plain FPN ablation aligned with retrieval-conditioned refinement.

This neck keeps the same post-fusion refinement operators as
RetrievalConditionedRefinement, but drops everything that requires retrieval:
additive prototype fusion ``F+V``, relation terms that use ``V``, and the
Top-1 confidence term ``a * s``.  Dynamic deformable alignment is also absent.
"""
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16

from ..builder import NECKS
from .fpn import FPN


class AblationSpatialRefinement(nn.Module):
    """RC-SA operators without retrieval-dependent inputs.

    Kept identical to RetrievalConditionedRefinement except:
    - relation input is only the refined feature (no V / U-V / U*V)
    - confidence term a*s is removed; learnable bias b is retained
    """

    def __init__(
        self,
        channels: int,
        gate_bias_init: float = 0.0,
    ):
        super().__init__()
        # Same CBR as RetrievalConditionedRefinement.feature_refinement.
        self.feature_refinement = ConvModule(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU', inplace=True),
        )

        # Same 1x1 reduction, but C->C because only refined is available.
        self.relation_reduction = ConvModule(
            channels,
            channels,
            kernel_size=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU', inplace=True),
        )
        # Same local 3x3 interaction.
        self.local_interaction = ConvModule(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU', inplace=True),
        )
        # Same 7x7 content gate head.
        self.content_gate = nn.Conv2d(
            channels, 1, kernel_size=7, padding=3)

        # Retain the learnable bias; drop confidence_scale because s is gone.
        self.gate_bias = nn.Parameter(
            torch.tensor(float(gate_bias_init)))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        refined = self.feature_refinement(feature)

        relation = self.relation_reduction(refined)
        relation = self.local_interaction(relation)

        content_logit = self.content_gate(relation)
        gate = torch.sigmoid(
            content_logit + self.gate_bias.to(dtype=content_logit.dtype))
        return refined * gate


@NECKS.register_module()
class PlainFPNSpatialRefinement(FPN):
    """Plain FPN + RC-SA-aligned spatial refinement without DCN/memory."""

    def __init__(
        self,
        *args,
        refinement_levels: Optional[List[int]] = None,
        gate_bias_init: float = 0.0,
        **kwargs,
    ):
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

        # Same replacement rule as RC-SA: refined * gate, no residual mix.
        for level in self.refinement_levels:
            if level < 0 or level >= len(laterals):
                raise IndexError(
                    f'refinement level {level} is out of range for '
                    f'{len(laterals)} laterals')
            laterals[level] = self.refinement_modules[str(level)](
                laterals[level])

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
