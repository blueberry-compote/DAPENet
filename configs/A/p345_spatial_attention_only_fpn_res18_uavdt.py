"""UAVDT ablation: RC-SA operators without DCN and without retrieval.

Keeps the same post-fusion refinement stack as the retrieval-conditioned
model, but removes additive prototype fusion and all V/s-dependent terms.
"""
_base_ = './drfpn_res18_uavdt.py'

custom_imports = dict(
    imports=[
        'mmdet.models.necks.plain_fpn_spatial_refinement',
    ],
    allow_failed_imports=False)

model = dict(
    type='GFL',
    neck=dict(
        _delete_=True,
        type='PlainFPNSpatialRefinement',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        start_level=1,
        end_level=-1,
        add_extra_convs='on_output',
        num_outs=5,
        refinement_levels=[0, 1, 2],
        # Same default as RC-SA gate_bias_init; confidence scale is unused.
        gate_bias_init=0.0))

work_dir = 'work_dirs/p345_spatial_attention_only_fpn_res18_uavdt'
