"""Similarity-gated retrieval followed by retrieval-conditioned RC-SA.

Fusion order:
    gate = sigmoid(a * s + b)
    F1   = F + gate * V
    out  = RC-SA(F1, V, s)

Does not modify the standalone similarity-gated or RC-SA modules; reuses their
bank / refinement operators by composition.
"""
from typing import Optional, Sequence

import torch
import torch.nn as nn

from .reorder_dc_dngeo_retrieval_conditioned_refinement import (
    RetrievalConditionedRefinement,
)
from .shared_pyramid_detail_memory import SharedPyramidDetailMemory
from .similarity_gated_shared_pyramid_detail_memory import (
    SimilarityGatedDetailMemoryBank,
)


class SimilarityGatedRetrievalConditionedMemory(SharedPyramidDetailMemory):
    """Shared P3/P4/P5 memory: gated injection then RC-SA refinement."""

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
        memory_gate_scale_init: float = 10.0,
        memory_gate_bias_init: float = -5.0,
        gate_scale_init: float = 1.0,
        gate_bias_init: float = 0.0,
    ):
        super().__init__(
            detail_in_channels=detail_in_channels,
            feat_channels=feat_channels,
            memory_size=memory_size,
            memory_dim=memory_dim,
            roi_size=roi_size,
            detail_stride=detail_stride,
            momentum=momentum,
            min_box_area=min_box_area,
            min_descriptor_norm=min_descriptor_norm,
            duplicate_threshold=duplicate_threshold,
            temperature=temperature,
            warmup_iters=warmup_iters,
            read_levels=read_levels,
        )
        self.bank = SimilarityGatedDetailMemoryBank(
            memory_size=memory_size,
            memory_dim=memory_dim,
            momentum=momentum,
            duplicate_threshold=duplicate_threshold,
            temperature=temperature,
        )
        # Soft spatial gate on prototype injection: F + gate(s) * V.
        self.memory_gate_scale = nn.Parameter(
            torch.tensor(float(memory_gate_scale_init)))
        self.memory_gate_bias = nn.Parameter(
            torch.tensor(float(memory_gate_bias_init)))

        self.refinement_modules = nn.ModuleDict({
            str(level): RetrievalConditionedRefinement(
                channels=feat_channels,
                gate_scale_init=gate_scale_init,
                gate_bias_init=gate_bias_init,
            )
            for level in self.read_levels
        })

    def _enhance_feature(
        self,
        feature: torch.Tensor,
        level: int,
    ) -> torch.Tensor:
        query = self.query_projection(feature)
        batch_size, channels, height, width = query.shape
        flat_query = query.permute(0, 2, 3, 1).reshape(-1, channels)
        retrieved, confidence = self.bank.read_with_confidence(flat_query)
        if retrieved is None:
            return feature

        retrieved = retrieved.view(
            batch_size, height, width, channels).permute(0, 3, 1, 2)
        retrieved_value = self.value_projection(retrieved)
        confidence = confidence.view(batch_size, 1, height, width)

        injection_gate = torch.sigmoid(
            self.memory_gate_scale.to(dtype=confidence.dtype) * confidence
            + self.memory_gate_bias.to(dtype=confidence.dtype))
        fused_feature = feature + injection_gate * retrieved_value
        return self.refinement_modules[str(level)](
            fused_feature, retrieved_value, confidence)

    def forward(self, laterals, detail_source=None, gt_bboxes=None):
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
            enhanced[level] = self._enhance_feature(
                enhanced[level], level)
        return enhanced
