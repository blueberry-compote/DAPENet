_base_ = [
    '../_base_/schedules/schedule_15e.py', '../_base_/default_runtime.py'
]

# ========================== 模型结构（完全保留原配置，确保权重衔接） ==========================
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
        type='FPNWithCAAU',
        in_channels=[64, 128, 256, 512],
        out_channels=512,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        caau_kwargs=dict(n_entities=8)),
    bbox_head=dict(
        type='GFLDYHead',
        num_classes=10,
        in_channels=512,
        stacked_convs=4,
        feat_channels=512,
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

# ========================== 数据配置（优化加载速度和增强策略） ==========================
dataset_type = 'UAVDataset'
data_root = '/root/autodl-tmp/VisDrone_Dataset_COCO_Format/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# 简化数据增强
albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.1,
        rotate_limit=45,
        interpolation=1,
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize',** img_norm_cfg),
    dict(type='Pad', size_divisor=32),
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
            dict(type='Normalize',** img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,  # 保持为2以适应显存
    workers_per_gpu=4,  # 提升数据加载线程
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

# ========================== 训练策略 ==========================
evaluation = dict(interval=2, metric='bbox')

# 优化器配置（关键修复：添加 _delete_=True）
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True,  # 必须添加此行，以解决类型冲突
    grad_clip=dict(max_norm=10, norm_type=2)
)

runner = dict(type='EpochBasedRunner', max_epochs=15)

find_unused_parameters = True