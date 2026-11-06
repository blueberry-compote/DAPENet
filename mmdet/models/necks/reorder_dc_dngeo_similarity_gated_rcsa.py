"""DDA -> similarity-gated retrieval -> RC-SA.

Extends the shared-memory DRFPN without modifying its source.  Independent FE
is disabled because RC-SA already provides post-fusion spatial refinement.
"""
from typing import Optional, Sequence

from ..builder import NECKS
from .reorder_dc_dngeo_shared_memory_light import (
    Reorder_DC_DNGeo_SharedMemory_Lite,
)
from .similarity_gated_retrieval_conditioned_memory import (
    SimilarityGatedRetrievalConditionedMemory,
)


@NECKS.register_module()
class Reorder_DC_DNGeo_SimilarityGatedRCSA(
        Reorder_DC_DNGeo_SharedMemory_Lite):
    """DDA neck with similarity-gated retrieval and RC-SA refinement."""

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
        memory_gate_scale_init: float = 10.0,
        memory_gate_bias_init: float = -5.0,
        gate_scale_init: float = 1.0,
        gate_bias_init: float = 0.0,
        **kwargs,
    ):
        # RC-SA owns post-fusion refinement; disable the original FE path.
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
            SimilarityGatedRetrievalConditionedMemory(
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
                memory_gate_scale_init=memory_gate_scale_init,
                memory_gate_bias_init=memory_gate_bias_init,
                gate_scale_init=gate_scale_init,
                gate_bias_init=gate_bias_init,
            )
        )
