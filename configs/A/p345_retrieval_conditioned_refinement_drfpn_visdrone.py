"""VisDrone: shared P3/P4/P5 memory with retrieval-conditioned refinement.

The original post-memory CBR + spatial attention is replaced by an integrated
module that keeps the order ``F1 = F + V -> CBR -> spatial selection``.  Its
spatial gate is conditioned on the retrieved value and Top-1 confidence.
"""
_base_ = './p345_shared_memory_drfpn_visdrone.py'

custom_imports = dict(
    imports=[
        'mmdet.models.detectors.gfl_shared_detail_memory',
        'mmdet.models.necks.'
        'reorder_dc_dngeo_retrieval_conditioned_refinement',
    ],
    allow_failed_imports=False)

model = dict(
    neck=dict(
        type='Reorder_DC_DNGeo_RetrievalConditionedRefinement',
        # confidence_logit = scale * top1_cosine_similarity + bias
        gate_scale_init=1.0,
        gate_bias_init=0.0))

work_dir = (
    'work_dirs/p345_retrieval_conditioned_refinement_drfpn_visdrone')
