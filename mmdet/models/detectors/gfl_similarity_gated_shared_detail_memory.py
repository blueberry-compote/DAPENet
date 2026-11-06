"""GFL detector for similarity-gated shared detail memory."""
from ..builder import DETECTORS
from .gfl_shared_detail_memory import GFLSharedDetailMemory


@DETECTORS.register_module()
class GFLSimilarityGatedSharedDetailMemory(GFLSharedDetailMemory):
    """GFL with GT-assisted writes to similarity-gated shared memory."""

