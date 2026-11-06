_base_ = [
    '../_base_/schedules/schedule_15e.py',
    '../_base_/default_runtime.py'
]

# 单阶段联合训练：从 DC+DNGeo 15e checkpoint 加载，直接训练 HFP_DC_DNGeo 15 个 epoch
# 用差异化 lr：HFP 用 1× lr，其余 backbone/DC/DNGeo/head 用 0.1× lr 微调适应。

total_epochs = 15
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[11, 14])

model = dict(
    type='GFL',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(
        type='HFP_DC_DNGeo',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        start_level=1,
        end_level=-1,
        add_extra_convs='on_output',
        num_outs=5,
        use_denoising=True,
        dcn_levels=[0, 1],
        denoising_levels=[0, 1, 2],
        # HFP：在 C2、C3 上启用，可学习 alpha，init_alpha 建议用你当前验证较好的值（如 0.15 或 0.20）
        # hfp_in_indices 是 HFP（高频增强）在FPN哪些阶段输入特征上启用的参数，(0, 1) 表示在 C2、C3（通常即 backbone 的第二、第三个输出）上应用 HFP 模块。
        hfp_in_indices=(0, 1),
        hfp_init_alpha=0.15,
        hfp_pool_k=16,
        hfp_cp_groups=1,
        hfp_highpass_temperature=2.0,
    ),
    bbox_head=dict(
        type='GFLHead',
        num_classes=10,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=16,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=500))

dataset_type = 'UAVDataset'
data_root = '/root/autodl-tmp/VisDrone/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

albu_train_transforms = [
    dict(type='ShiftScaleRotate', shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=1, p=0.5),
    dict(type='RandomBrightnessContrast', brightness_limit=[0.1, 0.3], contrast_limit=[0.1, 0.3], p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(type='RGBShift', r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1.0),
            dict(type='HueSaturationValue', hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0)
        ],
        p=0.1),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=64),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={'img': 'image', 'gt_bboxes': 'bboxes'},
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=64),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_UAVtrain.json',
            img_prefix=data_root + '/images/instances_UAVtrain/',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_UAVval.json',
        img_prefix=data_root + '/images/instances_UAVval/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_UAVval.json',
        img_prefix=data_root + '/images/instances_UAVval/images/',
        pipeline=test_pipeline))

evaluation = dict(interval=3, metric='bbox')

optimizer = dict(
    type='SGD',
    lr=0.0025,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_position_bias': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            # 差异化 lr：HFP 用 1×，其余用 0.1× 微调适应
            'backbone': dict(lr_mult=0.1),
            'neck.lateral_convs': dict(lr_mult=0.1),
            'neck.align_modules': dict(lr_mult=0.1),
            'neck.position_denoising_module': dict(lr_mult=0.1),
            'neck.fpn_convs': dict(lr_mult=0.1),
            'bbox_head': dict(lr_mult=0.1),
        }
    ))

optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=10, norm_type=2))

fp16 = dict(loss_scale=512.)

find_unused_parameters = True

checkpoint_config = dict(interval=1, max_keep_ckpts=3)

custom_hooks = [
    dict(type='NumClassCheckHook'),
    # 从 DC+DNGeo 15e 的 checkpoint 加载，strict=False 以跳过 HFP 模块缺失的权重
    dict(
        type='LoadCheckpointHook',
        path='work_dirs/gfl_res18_dc_0_1_dngeo_0_1_2_reorder_visdrone/epoch_15.pth',
        strict=False),
]

work_dir = 'work_dirs/gfl_res18_load_hfp_order_dc_choice_dngeo_reorder_choice_visdrone'

