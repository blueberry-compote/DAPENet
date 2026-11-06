"""Ablation: detail memory retrieval only (no DDA, no FE)."""
_base_ = './p345_shared_memory_drfpn_visdrone.py'

custom_imports = dict(
    imports=[
        'mmdet.models.detectors.gfl_shared_detail_memory',
        'mmdet.models.necks.reorder_dc_dngeo_shared_memory_retrieval_only',
    ],
    allow_failed_imports=False)

model = dict(
    neck=dict(
        type='Reorder_DC_DNGeo_SharedMemory_RetrievalOnly',
        dcn_levels=[],
        use_denoising=False))

work_dir = 'work_dirs/p345_shared_memory_retrieval_only_drfpn_visdrone'
