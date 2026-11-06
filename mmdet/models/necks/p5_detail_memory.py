"""P5-only detail memory used by the DDA-Memory-FE neck.

The memory is written from GT-aligned C2 details during training and read by
P5 in both training and inference. Memory updates are non-parametric EMA
updates; the query and fusion projections remain trainable.
"""
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.ops import roi_align


class DetailEncoder(nn.Module):
    """Compress a high-resolution backbone feature into detail descriptors."""

    def __init__(self, in_channels: int, memory_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvModule(
                in_channels,
                memory_dim,
                kernel_size=3,
                padding=1,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU', inplace=True)),
            nn.Conv2d(memory_dim, memory_dim, kernel_size=1))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.encoder(feature)


class DetailMemoryBank(nn.Module):
    """EMA-updated prototype memory with differentiable hard Top-1 reading."""

    def __init__(
        self,
        memory_size: int = 64,
        memory_dim: int = 128,
        momentum: float = 0.99,
        duplicate_threshold: float = 0.95,
        temperature: float = 0.07,
    ):
        super().__init__()
        if memory_size <= 0 or memory_dim <= 0:
            raise ValueError('memory_size and memory_dim must be positive')
        if not 0.0 <= momentum < 1.0:
            raise ValueError('momentum must be in [0, 1)')
        if temperature <= 0:
            raise ValueError('temperature must be positive')

        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.momentum = momentum
        self.duplicate_threshold = duplicate_threshold
        self.temperature = temperature

        memory = F.normalize(torch.randn(memory_size, memory_dim), dim=1)
        self.register_buffer('memory', memory)
        self.register_buffer(
            'valid_count', torch.zeros((), dtype=torch.long))
        self.register_buffer(
            'usage_count', torch.zeros(memory_size, dtype=torch.long))

    @torch.no_grad()
    def write(self, descriptors: torch.Tensor) -> int:
        """Write normalized descriptors with unified duplicate filtering.

        Whether the bank is full or not, a descriptor whose nearest existing
        prototype exceeds ``duplicate_threshold`` is discarded. Otherwise an
        empty slot is filled, or the nearest occupied slot is EMA-updated.
        """
        if descriptors.numel() == 0:
            return 0

        descriptors = F.normalize(descriptors.detach().float(), dim=1)
        written = 0
        for descriptor in descriptors:
            num_valid = int(self.valid_count.item())

            # Empty bank: occupy the first slot unconditionally.
            if num_valid == 0:
                self.memory[0].copy_(descriptor)
                self.valid_count.add_(1)
                written += 1
                continue

            # Always check diversity against currently valid prototypes.
            similarities = torch.mv(self.memory[:num_valid], descriptor)
            max_similarity, slot = similarities.max(dim=0)
            if max_similarity.item() >= self.duplicate_threshold:
                continue

            if num_valid < self.memory_size:
                self.memory[num_valid].copy_(descriptor)
                self.valid_count.add_(1)
            else:
                updated = (
                    self.momentum * self.memory[slot]
                    + (1.0 - self.momentum) * descriptor)
                self.memory[slot].copy_(F.normalize(updated, dim=-1))
                self.usage_count[slot].add_(1)
            written += 1
        return written

    def read(self, queries: torch.Tensor) -> Optional[torch.Tensor]:
        """Retrieve the most similar prototype for every query.

        Forward computation is exactly hard Top-1. A straight-through softmax
        supplies gradients to the query projection during backward.
        """
        num_valid = int(self.valid_count.item())
        if num_valid == 0:
            return None

        original_dtype = queries.dtype
        queries_float = F.normalize(queries.float(), dim=-1)
        memory = self.memory[:num_valid]
        similarities = torch.matmul(queries_float, memory.t())

        hard_indices = similarities.argmax(dim=-1)
        hard_weights = F.one_hot(
            hard_indices, num_classes=num_valid).to(similarities.dtype)
        soft_weights = F.softmax(similarities / self.temperature, dim=-1)
        weights = hard_weights - soft_weights.detach() + soft_weights
        retrieved = torch.matmul(weights, memory)

        with torch.no_grad():
            hits = torch.bincount(
                hard_indices.reshape(-1), minlength=num_valid)
            self.usage_count[:num_valid].add_(hits)
        return retrieved.to(dtype=original_dtype)


class P5DetailMemory(nn.Module):
    """Write C2 object details and inject retrieved prototypes into P5."""

    def __init__(
        self,
        detail_in_channels: int,
        p5_channels: int,
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
    ):
        super().__init__()
        self.roi_size = roi_size
        self.detail_stride = detail_stride
        self.min_box_area = min_box_area
        self.min_descriptor_norm = min_descriptor_norm
        self.warmup_iters = warmup_iters

        self.detail_encoder = DetailEncoder(detail_in_channels, memory_dim)
        self.query_projection = nn.Conv2d(
            p5_channels, memory_dim, kernel_size=1)
        self.value_projection = nn.Conv2d(
            memory_dim, p5_channels, kernel_size=1)
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

    def forward(
        self,
        p5: torch.Tensor,
        detail_source: Optional[torch.Tensor] = None,
        gt_bboxes: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
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
            return p5

        query = self.query_projection(p5)
        batch_size, channels, height, width = query.shape
        query = query.permute(0, 2, 3, 1).reshape(-1, channels)
        retrieved = self.bank.read(query)
        if retrieved is None:
            return p5

        retrieved = retrieved.view(
            batch_size, height, width, channels).permute(0, 3, 1, 2)
        return p5 + self.value_projection(retrieved)
