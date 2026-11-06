"""Plain FPN ablation that keeps only the detail-memory retrieval.

Counterpart of PlainFPNSpatialRefinement: everything except retrieval is
dropped.  There is no dynamic deformable alignment, no position denoising, no
post-fusion CBR, and no spatial/channel/confidence gating.  Selected laterals
are updated by additive prototype fusion ``F + V`` only.
"""
from typing import List, Optional, Sequence

import torch.nn.functional as F
from mmcv.runner import auto_fp16

from ..builder import NECKS
from .fpn import FPN
from .shared_pyramid_detail_memory import SharedPyramidDetailMemory


@NECKS.register_module()
class PlainFPNRetrievalOnly(FPN):
    """Plain FPN plus shared P3/P4/P5 detail-memory retrieval."""

    def __init__(
        self,
        *args,
        memory_size: int = 64,
        memory_dim: int = 128,
        memory_roi_size: int = 7,
        memory_detail_source_level: int = 0,
        memory_detail_stride: int = 4,
        memory_momentum: float = 0.99,
        memory_min_box_area: float = 16.0,
        memory_min_descriptor_norm: float = 1e-3,
        memory_duplicate_threshold: float = 0.95,
        memory_temperature: float = 0.07,
        memory_warmup_iters: int = 500,
        memory_read_levels: Optional[Sequence[int]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not 0 <= memory_detail_source_level < len(self.in_channels):
            raise ValueError(
                'memory_detail_source_level must index a backbone feature')

        self.memory_detail_source_level = memory_detail_source_level
        self.shared_detail_memory = SharedPyramidDetailMemory(
            detail_in_channels=self.in_channels[memory_detail_source_level],
            feat_channels=self.out_channels,
            memory_size=memory_size,
            memory_dim=memory_dim,
            roi_size=memory_roi_size,
            detail_stride=memory_detail_stride,
            momentum=memory_momentum,
            min_box_area=memory_min_box_area,
            min_descriptor_norm=memory_min_descriptor_norm,
            duplicate_threshold=memory_duplicate_threshold,
            temperature=memory_temperature,
            warmup_iters=memory_warmup_iters,
            read_levels=memory_read_levels)

    def get_memory_stats(self):
        """Return lightweight runtime statistics for debugging/logging."""
        bank = self.shared_detail_memory.bank
        return dict(
            valid_count=int(bank.valid_count.item()),
            candidates=self.shared_detail_memory.last_num_candidates,
            written=self.shared_detail_memory.last_num_written,
            read_levels=list(self.shared_detail_memory.read_levels))

    @auto_fp16()
    def forward(self, inputs, gt_bboxes: Optional[List] = None):
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

        # Write from C2 with GT boxes, then read/add prototypes on P3/P4/P5.
        laterals = self.shared_detail_memory(
            laterals,
            detail_source=inputs[self.memory_detail_source_level],
            gt_bboxes=gt_bboxes)

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
