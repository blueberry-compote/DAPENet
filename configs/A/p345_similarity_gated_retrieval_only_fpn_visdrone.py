"""VisDrone ablation: similarity-gated detail-memory retrieval only.

Plain FPN top-down fusion plus gated prototype fusion ``F + gate * V`` on
P3/P4/P5, where ``gate = sigmoid(scale * top1_cosine + bias)``.  No DDA, no
position denoising, and no independent FE attention.  Memory / gate
hyper-parameters match p345_similarity_gated_shared_memory_drfpn_visdrone so
the comparison isolates this retrieval path.
"""
_base_ = './p345_similarity_gated_shared_memory_drfpn_visdrone.py'

custom_imports = dict(
    imports=[
        'mmdet.models.detectors.gfl_similarity_gated_shared_detail_memory',
        'mmdet.models.necks.plain_fpn_similarity_gated_retrieval_only',
    ],
    allow_failed_imports=False)

model = dict(
    neck=dict(
        _delete_=True,
        type='PlainFPNSimilarityGatedRetrievalOnly',
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
        memory_warmup_iters=500,
        memory_gate_scale_init=10.0,
        memory_gate_bias_init=-5.0))

work_dir = 'work_dirs/p345_similarity_gated_retrieval_only_fpn_visdrone'
