"""UAVDT ablation #1: no DDA, similarity-gated retrieval + old FE (CBR+SA).

Plain FPN top-down fusion, then on P3/P4/P5:
    F1  = F + sigmoid(a * s + b) * V
    out = OldFE(F1)   # CBR -> 7x7 SA -> residual; no channel attn
"""
_base_ = './drfpn_res18_uavdt.py'

custom_imports = dict(
    imports=[
        'mmdet.models.detectors.gfl_similarity_gated_shared_detail_memory',
        'mmdet.models.necks.plain_fpn_similarity_gated_old_fe',
    ],
    allow_failed_imports=False)

model = dict(
    type='GFLSimilarityGatedSharedDetailMemory',
    neck=dict(
        _delete_=True,
        type='PlainFPNSimilarityGatedOldFE',
        in_channels=[64, 128, 256, 512],
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
        memory_gate_bias_init=-5.0,
        denoising_levels=[0, 1, 2],
        use_spatial_attn=True,
        use_channel_attn=False))

work_dir = 'work_dirs/p345_similarity_gated_old_fe_fpn_res18_uavdt'
