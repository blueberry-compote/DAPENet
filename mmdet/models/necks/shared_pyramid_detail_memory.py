"""Shared-projection detail memory for P3/P4/P5.

Write path is identical to the P5-only module: C2 + GT ROI descriptors are
stored in one EMA memory bank. Read path injects retrieved prototypes into
selected FPN laterals with a single shared query/value head.
"""
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
from mmcv.ops import roi_align

from .p5_detail_memory import DetailEncoder, DetailMemoryBank


class SharedPyramidDetailMemory(nn.Module):
    """Write once from C2, read on multiple FPN levels with shared projections."""

    def __init__(
        self,
        detail_in_channels: int,
        feat_channels: int,
        memory_size: int = 64,
        memory_dim: int = 128,
        roi_size: int = 7,
        detail_stride: int = 4,
        momentum: float = 0.99,
        min_box_area: float = 16.0,
        min_descriptor_norm: float = 1e-3,
        duplicate_threshold: float = 0.95,
        temperature: float = 0.07,
        warmup_iters: int = 500,
        read_levels: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.roi_size = roi_size
        self.detail_stride = detail_stride
        self.min_box_area = min_box_area
        self.min_descriptor_norm = min_descriptor_norm
        self.warmup_iters = warmup_iters
        self.read_levels = (
            [0, 1, 2] if read_levels is None else list(read_levels))

        self.detail_encoder = DetailEncoder(detail_in_channels, memory_dim)
        # One shared query/value head for all selected pyramid levels.
        self.query_projection = nn.Conv2d(
            feat_channels, memory_dim, kernel_size=1)
        self.value_projection = nn.Conv2d(
            memory_dim, feat_channels, kernel_size=1)
        self.bank = DetailMemoryBank(
            memory_size=memory_size,
            memory_dim=memory_dim,
            momentum=momentum,
            duplicate_threshold=duplicate_threshold,
            temperature=temperature)
        self.register_buffer(
            'num_updates', torch.zeros((), dtype=torch.long))

        self.last_num_candidates = 0
        self.last_num_written = 0

    def _build_rois(
        self,
        gt_bboxes: List[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ):
        rois = []
        areas = []
        for batch_index, boxes in enumerate(gt_bboxes):
            if boxes is None or boxes.numel() == 0:
                continue
            boxes = boxes.to(device=device, dtype=dtype)
            wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0)
            box_areas = wh[:, 0] * wh[:, 1]
            batch_indices = boxes.new_full(
                (boxes.size(0), 1), float(batch_index))
            rois.append(torch.cat([batch_indices, boxes], dim=1))
            areas.append(box_areas)

        if not rois:
            return None, None
        return torch.cat(rois, dim=0), torch.cat(areas, dim=0)

    def _extract_descriptors(
        self,
        detail_feature: torch.Tensor,
        gt_bboxes: List[torch.Tensor],
    ) -> torch.Tensor:
        rois, areas = self._build_rois(
            gt_bboxes, detail_feature.device, detail_feature.dtype)
        if rois is None:
            return detail_feature.new_empty((0, self.bank.memory_dim))

        pooled = roi_align(
            detail_feature,
            rois,
            (self.roi_size, self.roi_size),
            1.0 / self.detail_stride,
            0,
            'avg',
            True)
        descriptors = pooled.mean(dim=(2, 3))

        descriptor_norms = descriptors.float().norm(p=2, dim=1)
        keep = (
            (areas.float() >= self.min_box_area)
            & (descriptor_norms >= self.min_descriptor_norm))
        return descriptors[keep]

    def _enhance_feature(self, feature: torch.Tensor) -> torch.Tensor:
        query = self.query_projection(feature)
        batch_size, channels, height, width = query.shape
        query = query.permute(0, 2, 3, 1).reshape(-1, channels)
        retrieved = self.bank.read(query)
        if retrieved is None:
            return feature

        retrieved = retrieved.view(
            batch_size, height, width, channels).permute(0, 3, 1, 2)
        return feature + self.value_projection(retrieved)

    def forward(
        self,
        laterals: List[torch.Tensor],
        detail_source: Optional[torch.Tensor] = None,
        gt_bboxes: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        if self.training:
            self.num_updates.add_(1)
            self.last_num_candidates = 0
            self.last_num_written = 0
            if detail_source is not None and gt_bboxes is not None:
                detail_feature = self.detail_encoder(detail_source)
                descriptors = self._extract_descriptors(
                    detail_feature, gt_bboxes)
                self.last_num_candidates = descriptors.size(0)
                self.last_num_written = self.bank.write(descriptors)

        if (
            self.training
            and int(self.num_updates.item()) <= self.warmup_iters
        ):
            return laterals

        enhanced = list(laterals)
        for level in self.read_levels:
            if level < 0 or level >= len(enhanced):
                raise IndexError(
                    f'read level {level} is out of range for '
                    f'{len(enhanced)} laterals')
            enhanced[level] = self._enhance_feature(enhanced[level])
        return enhanced
