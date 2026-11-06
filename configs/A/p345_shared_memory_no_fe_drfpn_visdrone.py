"""Ablation: shared memory + DDA, entire FE removed (CBR + spatial + channel)."""
_base_ = './p345_shared_memory_drfpn_visdrone.py'

custom_imports = dict(
    imports=[
        'mmdet.models.detectors.gfl_shared_detail_memory',
        'mmdet.models.necks.reorder_dc_dngeo_shared_memory_attn_ablation',
    ],
    allow_failed_imports=False)

model = dict(
    neck=dict(
        type='Reorder_DC_DNGeo_SharedMemory_AttnAblation',
        use_spatial_attn=False,
        use_channel_attn=False))

work_dir = 'work_dirs/p345_shared_memory_no_fe_drfpn_visdrone'
