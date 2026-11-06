"""VisDrone: GFL + DDA + shared P3/P4/P5 detail memory + FE.

Inherits the existing VisDrone DRFPN experiment. Only detector/neck types and
shared-memory options are overridden. Existing files are left untouched.
"""
_base_ = './drfpn_visdrone.py'

custom_imports = dict(
    imports=[
        'mmdet.models.detectors.gfl_shared_detail_memory',
        'mmdet.models.necks.reorder_dc_dngeo_shared_memory_light',
    ],
    allow_failed_imports=False)

model = dict(
    type='GFLSharedDetailMemory',
    neck=dict(
        type='Reorder_DC_DNGeo_SharedMemory_Lite',
        # Memory shape: 64 slots x 128-D, shared across P3/P4/P5.
        memory_size=64,
        memory_dim=128,
        # C2 (stride 4) provides GT-aligned detail descriptors.
        memory_detail_source_level=0,
        memory_detail_stride=4,
        memory_roi_size=7,
        # EMA write and quality filtering.
        memory_momentum=0.99,
        memory_min_box_area=16.0,
        memory_min_descriptor_norm=1e-3,
        memory_duplicate_threshold=0.95,
        # Shared query/value projections; read P3, P4 and P5.
        memory_read_levels=[0, 1, 2],
        memory_temperature=0.07,
        memory_warmup_iters=500))

work_dir = 'work_dirs/p345_shared_memory_drfpn_visdrone'
