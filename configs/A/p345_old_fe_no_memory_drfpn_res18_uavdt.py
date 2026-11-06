"""UAVDT ablation #4: DDA + old FE (CBR+SA), no retrieval.

Keeps dynamic deformable alignment from the UAVDT DRFPN base. After top-down
fusion, applies PositionDenoising-style FE on P3/P4/P5 with spatial attention
only (no channel attention). No detail-memory retrieval.
"""
_base_ = './drfpn_res18_uavdt.py'

custom_imports = dict(
    imports=[
        'mmdet.models.necks.reorder_dc_dngeo_attn_ablation_no_memory',
    ],
    allow_failed_imports=False)

model = dict(
    type='GFL',
    neck=dict(
        type='Reorder_DC_DNGeo_AttnAblation_NoMemory',
        # Keep DDA from the UAVDT DRFPN base (dcn_levels=[0, 1]).
        denoising_levels=[0, 1, 2],
        use_spatial_attn=True,
        use_channel_attn=False))

work_dir = 'work_dirs/p345_old_fe_no_memory_drfpn_res18_uavdt'
