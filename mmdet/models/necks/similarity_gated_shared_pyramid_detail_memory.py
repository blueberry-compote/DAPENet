"""Shared pyramid detail memory with retrieval-confidence spatial gating.

Each spatial query still retrieves one hard Top-1 memory prototype.  The
corresponding maximum cosine similarity is converted into a soft spatial gate
that controls how strongly the retrieved value is injected into the feature.
"""
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .p5_detail_memory import DetailMemoryBank
from .shared_pyramid_detail_memory import SharedPyramidDetailMemory


class SimilarityGatedDetailMemoryBank(DetailMemoryBank):
    """Detail memory bank that also returns Top-1 matching confidence."""

    def read_with_confidence(self, queries: torch.Tensor):
        """Retrieve hard Top-1 values and their maximum cosine similarities."""
        num_valid = int(self.valid_count.item())
        if num_valid == 0:
            return None, None

        original_dtype = queries.dtype
        queries_float = F.normalize(queries.float(), dim=-1)
        memory = self.memory[:num_valid]
        similarities = torch.matmul(queries_float, memory.t())

        confidence, hard_indices = similarities.max(dim=-1)
        hard_weights = F.one_hot(
            hard_indices, num_classes=num_valid).to(similarities.dtype)
        soft_weights = F.softmax(
            similarities / self.temperature, dim=-1)
        # Hard Top-1 in the forward pass, soft retrieval in the backward pass.
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


class SimilarityGatedSharedPyramidDetailMemory(
        SharedPyramidDetailMemory):
    """Shared P3/P4/P5 memory with similarity-guided spatial injection."""

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
        gate_scale_init: float = 10.0,
        gate_bias_init: float = -5.0,
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
        # sigmoid(scale * similarity + bias) learns the gate sharpness and
        # threshold.  The defaults correspond to an initial threshold of 0.5.
        self.gate_scale = nn.Parameter(torch.tensor(float(gate_scale_init)))
        self.gate_bias = nn.Parameter(torch.tensor(float(gate_bias_init)))

    def _enhance_feature(self, feature: torch.Tensor) -> torch.Tensor:
        query = self.query_projection(feature)
        batch_size, channels, height, width = query.shape
        flat_query = query.permute(0, 2, 3, 1).reshape(-1, channels)
        retrieved, confidence = self.bank.read_with_confidence(flat_query)
        if retrieved is None:
            return feature

        retrieved = retrieved.view(
            batch_size, height, width, channels).permute(0, 3, 1, 2)
        confidence = confidence.view(batch_size, 1, height, width)
        gate = torch.sigmoid(
            self.gate_scale.to(dtype=confidence.dtype) * confidence
            + self.gate_bias.to(dtype=confidence.dtype))
        return feature + gate * self.value_projection(retrieved)
