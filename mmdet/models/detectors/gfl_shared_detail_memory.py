"""GFL detector for the shared-projection multi-level detail memory neck."""
from ..builder import DETECTORS
from .gfl_p5_detail_memory import GFLP5DetailMemory


@DETECTORS.register_module()
class GFLSharedDetailMemory(GFLP5DetailMemory):
    """Same GT-to-neck training hook as GFLP5DetailMemory, separate registry name."""
