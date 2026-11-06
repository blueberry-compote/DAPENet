"""VisDrone ablation: detail-memory retrieval only, nothing else.

Plain FPN top-down fusion plus additive prototype fusion ``F + V`` on
P3/P4/P5.  No DDA, no position denoising, no post-fusion CBR, and no
spatial/channel/confidence gating.  Memory hyper-parameters match
p345_shared_memory_drfpn_visdrone so the comparison isolates retrieval.
"""
_base_ = './p345_shared_memory_drfpn_visdrone.py'

custom_imports = dict(
    imports=[
        'mmdet.models.detectors.gfl_shared_detail_memory',
        'mmdet.models.necks.plain_fpn_retrieval_only',
    ],
    allow_failed_imports=False)

model = dict(
    neck=dict(
        _delete_=True,
        type='PlainFPNRetrievalOnly',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        end_level=-1,
        add_extra_convs='on_output',
        num_outs=5,
        memory_size=64,
        memory_dim=128,
        memory_detail_source_level=0,
        memory_detail_stride=4,
        memory_roi_size=7,
        memory_momentum=0.99,
        memory_min_box_area=16.0,
        memory_min_descriptor_norm=1e-3,
        memory_duplicate_threshold=0.95,
        memory_read_levels=[0, 1, 2],
        memory_temperature=0.07,
        memory_warmup_iters=500))

work_dir = 'work_dirs/p345_retrieval_only_fpn_visdrone'
