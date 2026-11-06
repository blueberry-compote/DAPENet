"""VisDrone: GFL + DDA + P5 detail memory + FE.

This config inherits the existing VisDrone experiment unchanged except for
the new detector/neck types and P5-memory parameters.
"""
_base_ = './drfpn_visdrone.py'

custom_imports = dict(
    imports=[
        'mmdet.models.detectors.gfl_p5_detail_memory',
        'mmdet.models.necks.reorder_dc_dngeo_p5_memory_light',
    ],
    allow_failed_imports=False)

model = dict(
    type='GFLP5DetailMemory',
    neck=dict(
        type='Reorder_DC_DNGeo_P5Memory_Lite',
        # Memory shape: 64 slots, each represented by a 128-D vector.
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
        # Read only P5 using differentiable hard Top-1 retrieval.
        memory_temperature=0.07,
        memory_warmup_iters=500))

work_dir = 'work_dirs/p5_memory_drfpn_visdrone'
