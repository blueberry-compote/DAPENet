_base_ = [
    '../_base_/schedules/schedule_15e.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='GFL',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=0,  # 改进1: 从1改为0，允许所有层参与训练，提升特征学习能力
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(
        type='DC_FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        start_level=1,
        end_level=-1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='GFLHead',
        num_classes=10,
        in_channels=256,
        stacked_convs=6,  # 改进2: 从4增加到6，增强特征提取能力
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],          # 单 anchor 版本
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
    # training and testing settings
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

# dataset settings
dataset_type = 'UAVDataset'
data_root = '/root/autodl-tmp/VisDrone/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# 改进3: 增强数据增强策略，提高概率和强度
albu_train_transforms = [
    dict(type='ShiftScaleRotate', shift_limit=0.1, scale_limit=0.15, rotate_limit=45, interpolation=1, p=0.6),  # 增强
    dict(type='RandomBrightnessContrast', brightness_limit=[0.15, 0.35], contrast_limit=[0.15, 0.35], p=0.3),  # 增强
    dict(
        type='OneOf',
        transforms=[
            dict(type='RGBShift', r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0),  # 增强
            dict(type='HueSaturationValue', hue_shift_limit=25, sat_shift_limit=35, val_shift_limit=25, p=1.0)  # 增强
        ],
        p=0.15),  # 提高概率
    dict(type='JpegCompression', quality_lower=80, quality_upper=95, p=0.25),  # 增强
    dict(type='ChannelShuffle', p=0.15),  # 提高概率
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=5, p=1.0),  # 增强
            dict(type='MedianBlur', blur_limit=5, p=1.0),  # 增强
            dict(type='MotionBlur', blur_limit=5, p=1.0)  # 新增
        ],
        p=0.15),  # 提高概率
    dict(type='GaussNoise', var_limit=(10.0, 50.0), p=0.1),  # 新增噪声增强
]

# 改进4: 添加多尺度训练
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 800), (1600, 960), (1200, 720)],  # 多尺度训练
        multiscale_mode='value',
        keep_ratio=True),
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
        keymap={
            'img': 'image',
            # 'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

# 改进5: 启用测试时增强（TTA）
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1333, 800), (1600, 960), (1200, 720)],  # 多尺度测试
        flip=True,  # 启用翻转
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
            'norm': dict(decay_mult=0.)}
    ))

optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=10, norm_type=2))

# 改进6: 使用余弦退火学习率策略（保持15轮训练）
lr_config = dict(
    policy='CosineAnnealing',  # 从step改为CosineAnnealing
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    min_lr=0.00001)  # 最小学习率

fp16 = dict(loss_scale=512.)    

find_unused_parameters = True

# checkpoint配置：只保留最近3个checkpoint文件
checkpoint_config = dict(
    interval=1,           # 每个epoch保存一次
    max_keep_ckpts=3      # 只保留最近3个checkpoint，自动删除旧的
)