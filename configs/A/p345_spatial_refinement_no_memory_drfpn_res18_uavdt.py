"""UAVDT ablation: DDA + RC-SA operators, no retrieval.

Keeps dynamic deformable alignment.  After top-down fusion, applies the same
CBR + spatial-gate stack as AblationSpatialRefinement on P3/P4/P5, without
memory bank, prototype injection, or V/s-conditioned terms.
"""
_base_ = './drfpn_res18_uavdt.py'

custom_imports = dict(
    imports=[
        'mmdet.models.necks.reorder_dc_dngeo_spatial_refinement_no_memory',
    ],
    allow_failed_imports=False)

model = dict(
    type='GFL',
    neck=dict(
        type='Reorder_DC_DNGeo_SpatialRefinement_NoMemory',
        # Keep DDA from the UAVDT DRFPN base (dcn_levels=[0, 1]).
        refinement_levels=[0, 1, 2],
        gate_bias_init=0.0))

work_dir = 'work_dirs/p345_spatial_refinement_no_memory_drfpn_res18_uavdt'
