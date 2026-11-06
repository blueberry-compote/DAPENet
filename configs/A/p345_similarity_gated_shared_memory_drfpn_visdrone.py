"""VisDrone: DDA + similarity-gated shared P3/P4/P5 memory (no FE).

The retrieval Top-1 cosine similarity is used as a soft spatial gate during
memory fusion, replacing the entire FE block (CBR + spatial + channel attn).
"""
_base_ = './p345_shared_memory_drfpn_visdrone.py'

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
        # gate = sigmoid(scale * top1_cosine_similarity + bias)
        # These defaults initialize the soft matching threshold at 0.5.
        memory_gate_scale_init=10.0,
        memory_gate_bias_init=-5.0,
        # Disable entire FE (CBR + SA + CA); spatial gating lives in retrieval.
        use_spatial_attn=False,
        use_channel_attn=False))

work_dir = (
    'work_dirs/p345_similarity_gated_shared_memory_drfpn_visdrone')
