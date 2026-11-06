"""UAVDT ablation #3: no DDA, no retrieval, old FE only (CBR+SA).

Plain FPN top-down fusion, then PositionDenoising-style FE on P3/P4/P5:
CBR -> Conv7x7 sigmoid spatial gate -> residual blend. No channel attn.
"""
_base_ = './drfpn_res18_uavdt.py'

custom_imports = dict(
    imports=[
        'mmdet.models.necks.plain_fpn_old_fe',
    ],
    allow_failed_imports=False)

model = dict(
    type='GFL',
    neck=dict(
        _delete_=True,
        type='PlainFPNOldFE',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        start_level=1,
        end_level=-1,
        add_extra_convs='on_output',
        num_outs=5,
        denoising_levels=[0, 1, 2],
        use_spatial_attn=True,
        use_channel_attn=False))

work_dir = 'work_dirs/p345_old_fe_only_fpn_res18_uavdt'
