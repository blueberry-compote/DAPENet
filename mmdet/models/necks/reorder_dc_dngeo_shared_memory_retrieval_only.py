"""Shared detail memory retrieval only (no DDA, no FE).

Extends Reorder_DC_DNGeo_SharedMemory_Lite without modifying its source.
Top-down fusion uses plain interpolation + addition; memory write/read on
P3/P4/P5 is kept; the position-denoising FE block is disabled.
"""
from ..builder import NECKS
from .reorder_dc_dngeo_shared_memory_light import (
    Reorder_DC_DNGeo_SharedMemory_Lite,
)


@NECKS.register_module()
class Reorder_DC_DNGeo_SharedMemory_RetrievalOnly(
        Reorder_DC_DNGeo_SharedMemory_Lite):
    """FPN neck with shared detail memory only."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('dcn_levels', [])
        kwargs['use_denoising'] = False
        super().__init__(*args, **kwargs)
