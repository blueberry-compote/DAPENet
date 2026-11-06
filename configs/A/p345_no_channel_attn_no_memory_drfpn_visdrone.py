"""Ablation: no channel attention, no detail memory (DDA + CBR + SA only)."""
_base_ = './p345_shared_memory_no_channel_attn_drfpn_visdrone.py'

custom_imports = dict(
    imports=[
        'mmdet.models.necks.reorder_dc_dngeo_attn_ablation_no_memory',
    ],
    allow_failed_imports=False)

model = dict(
    type='GFL',
    neck=dict(
        _delete_=True,
        type='Reorder_DC_DNGeo_AttnAblation_NoMemory',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        end_level=-1,
        add_extra_convs='on_output',
        num_outs=5,
        dcn_levels=[0, 1],
        denoising_levels=[0, 1, 2],
        use_spatial_attn=True,
        use_channel_attn=False))

work_dir = 'work_dirs/p345_no_channel_attn_no_memory_drfpn_visdrone'
