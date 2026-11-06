"""UAVDT ablation: DDA + similarity-gated retrieval, no RC-SA / FE.

After DDA top-down fusion, on P3/P4/P5:
    F_out = F + sigmoid(a * s + b) * V
No independent FE and no RC-SA; spatial gating lives only in retrieval.
Hyper-parameters match the VisDrone similarity-gated shared-memory setup.
"""
_base_ = './drfpn_res18_uavdt.py'

custom_imports = dict(
    imports=[
        'mmdet.models.detectors.gfl_similarity_gated_shared_detail_memory',
        'mmdet.models.necks.reorder_dc_dngeo_similarity_gated_shared_memory',
    ],
    allow_failed_imports=False)

model = dict(
    type='GFLSimilarityGatedSharedDetailMemory',
    neck=dict(
        type='Reorder_DC_DNGeo_SimilarityGatedSharedMemory',
        # Keep DDA from the UAVDT DRFPN base (dcn_levels=[0, 1]).
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
        # gate = sigmoid(scale * top1_cosine + bias); init threshold ~0.5
        memory_gate_scale_init=10.0,
        memory_gate_bias_init=-5.0,
        # Disable entire FE (CBR + SA + CA); no RC-SA either.
        use_spatial_attn=False,
        use_channel_attn=False))

work_dir = (
    'work_dirs/p345_similarity_gated_shared_memory_drfpn_res18_uavdt')
