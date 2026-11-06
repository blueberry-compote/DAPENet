"""Shared-memory DRFPN with retrieval-conditioned spatial refinement.

The refinement is performed after additive memory fusion.  Unlike the
original spatial attention, its gate requires both the retrieved prototype
and its Top-1 matching confidence, so it cannot operate without retrieval.
"""
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from ..builder import NECKS
from .p5_detail_memory import DetailMemoryBank
from .reorder_dc_dngeo_shared_memory_light import (
    Reorder_DC_DNGeo_SharedMemory_Lite,
)
from .shared_pyramid_detail_memory import SharedPyramidDetailMemory


class ConfidenceDetailMemoryBank(DetailMemoryBank):
    """Hard Top-1 memory reader that also returns matching confidence."""

    def read_with_confidence(self, queries: torch.Tensor):
        num_valid = int(self.valid_count.item())
        if num_valid == 0:
            return None, None

        original_dtype = queries.dtype
        normalized_queries = F.normalize(queries.float(), dim=-1)
        memory = self.memory[:num_valid]
        similarities = torch.matmul(normalized_queries, memory.t())

        confidence, hard_indices = similarities.max(dim=-1)
        hard_weights = F.one_hot(
            hard_indices, num_classes=num_valid).to(similarities.dtype)
        soft_weights = F.softmax(
            similarities / self.temperature, dim=-1)
        weights = hard_weights - soft_weights.detach() + soft_weights
        retrieved = torch.matmul(weights, memory)

        with torch.no_grad():
            hits = torch.bincount(
                hard_indices.reshape(-1), minlength=num_valid)
            self.usage_count[:num_valid].add_(hits)

        return (
            retrieved.to(dtype=original_dtype),
            confidence.to(dtype=original_dtype),
        )


class RetrievalConditionedRefinement(nn.Module):
    """Refine a fused feature using retrieval-dependent spatial selection."""

    def __init__(
        self,
        channels: int,
        gate_scale_init: float = 1.0,
        gate_bias_init: float = 0.0,
    ):
        super().__init__()
        # This CBR corresponds to the original post-fusion feature enhancer.
        self.feature_refinement = ConvModule(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU', inplace=True),
        )

        relation_channels = 4 * channels
        self.relation_reduction = ConvModule(
            relation_channels,
            channels,
            kernel_size=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU', inplace=True),
        )
        self.local_interaction = ConvModule(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU', inplace=True),
        )
        self.content_gate = nn.Conv2d(
            channels, 1, kernel_size=7, padding=3)

        # Per-level affine calibration of Top-1 cosine similarity.
        self.confidence_scale = nn.Parameter(
            torch.tensor(float(gate_scale_init)))
        self.gate_bias = nn.Parameter(
            torch.tensor(float(gate_bias_init)))

    def forward(
        self,
        fused_feature: torch.Tensor,
        retrieved_value: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor:
        refined = self.feature_refinement(fused_feature)

        relation = torch.cat(
            [
                refined,
                retrieved_value,
                refined - retrieved_value,
                refined * retrieved_value,
            ],
            dim=1,
        )
        relation = self.relation_reduction(relation)
        relation = self.local_interaction(relation)

        content_logit = self.content_gate(relation)
        confidence_logit = (
            self.confidence_scale.to(dtype=confidence.dtype) * confidence
            + self.gate_bias.to(dtype=confidence.dtype)
        )
        gate = torch.sigmoid(content_logit + confidence_logit)
        return refined * gate


class RetrievalConditionedSharedPyramidMemory(
        SharedPyramidDetailMemory):
    """Add retrieved values first, then apply retrieval-conditioned FE."""

    def __init__(
        self,
        *args,
        gate_scale_init: float = 1.0,
        gate_bias_init: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.bank = ConfidenceDetailMemoryBank(
            memory_size=self.bank.memory_size,
            memory_dim=self.bank.memory_dim,
            momentum=self.bank.momentum,
            duplicate_threshold=self.bank.duplicate_threshold,
            temperature=self.bank.temperature,
        )

        self.refinement_modules = nn.ModuleDict({
            str(level): RetrievalConditionedRefinement(
                channels=self.value_projection.out_channels,
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

        # Preserve the original order: additive memory fusion precedes FE.
        fused_feature = feature + retrieved_value
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


@NECKS.register_module()
class Reorder_DC_DNGeo_RetrievalConditionedRefinement(
        Reorder_DC_DNGeo_SharedMemory_Lite):
    """Shared-memory neck with retrieval-conditioned post-fusion refinement."""

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
        gate_scale_init: float = 1.0,
        gate_bias_init: float = 0.0,
        **kwargs,
    ):
        # The retrieval-conditioned module contains CBR and spatial selection;
        # disable the independent original FE to avoid applying enhancement twice.
        kwargs['use_denoising'] = False
        super().__init__(
            *args,
            memory_size=memory_size,
            memory_dim=memory_dim,
            memory_roi_size=memory_roi_size,
            memory_detail_source_level=memory_detail_source_level,
            memory_detail_stride=memory_detail_stride,
            memory_momentum=memory_momentum,
            memory_min_box_area=memory_min_box_area,
            memory_min_descriptor_norm=memory_min_descriptor_norm,
            memory_duplicate_threshold=memory_duplicate_threshold,
            memory_temperature=memory_temperature,
            memory_warmup_iters=memory_warmup_iters,
            memory_read_levels=memory_read_levels,
            **kwargs,
        )

        self.shared_detail_memory = (
            RetrievalConditionedSharedPyramidMemory(
                detail_in_channels=self.in_channels[
                    memory_detail_source_level],
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
                read_levels=memory_read_levels,
                gate_scale_init=gate_scale_init,
                gate_bias_init=gate_bias_init,
            )
        )
