"""DDA -> shared P3/P4/P5 detail memory -> FE neck.

Extends Reorder_DC_DNGeo_Lite without modifying its source. After top-down
DDA, a shared-projection memory module writes from C2 and reads on selected
laterals (default P3/P4/P5), then the original FE path continues.
"""
from typing import List, Optional, Sequence

import torch.nn.functional as F
from mmcv.runner import auto_fp16

from ..builder import NECKS
from .reorder_dc_dngeo_light import Reorder_DC_DNGeo_Lite
from .shared_pyramid_detail_memory import SharedPyramidDetailMemory


@NECKS.register_module()
class Reorder_DC_DNGeo_SharedMemory_Lite(Reorder_DC_DNGeo_Lite):
    """Reorder DC-DNGeo neck with shared-projection multi-level detail memory."""

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

    def _build_topdown_memory_and_outputs(
        self,
        laterals,
        inputs,
        gt_bboxes: Optional[List] = None,
    ):
        pre_topdown_laterals = [l.clone() for l in laterals]
        used_backbone_levels = len(laterals)

        # 1. Original top-down fusion and DDA.
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

        # 2. Shared memory: write from C2, read/fuse on P3/P4/P5 by default.
        laterals = self.shared_detail_memory(
            laterals,
            detail_source=inputs[self.memory_detail_source_level],
            gt_bboxes=gt_bboxes)

        # 3. Original post-top-down feature enhancement.
        if self.denoising_after_topdown and self.use_denoising:
            denoised_laterals = self.position_denoising_module(
                laterals, levels_to_denoise=self.denoising_levels)
            for i in self.denoising_levels:
                if i < len(laterals):
                    original = laterals[i]
                    laterals[i] = original + self.denoising_weight * (
                        denoised_laterals[i] - original)

        # 4. Original P3-P5 output convolutions.
        outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels)
        ]

        # 5. Original extra P6/P7 output construction.
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
                for i in range(
                    used_backbone_levels + 1, self.num_outs
                ):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        return tuple(outs), pre_topdown_laterals

    @auto_fp16()
    def forward(self, inputs, gt_bboxes=None):
        assert len(inputs) == len(self.in_channels)
        laterals = self._build_laterals_with_denoising(inputs)
        fpn_outs, _ = self._build_topdown_memory_and_outputs(
            laterals, inputs, gt_bboxes=gt_bboxes)
        return fpn_outs
