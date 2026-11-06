_base_ = [
    '../_base_/schedules/schedule_15e.py', '../_base_/default_runtime.py'
]

# ======================================================
# 🚀 RTX 4090 训练加速优化版 RetinaNet + GuidedFPN
# ======================================================

model = dict(
    type='RetinaNet',
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
    
    # ==== 自定义 GuidedFPN ====
    neck=dict(
        type='GuidedFPN',
        in_channels=[64, 128, 256, 512],
        out_channels=512,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5,
        guided_lsk_cfg=dict(
            reduction=2,
            use_channel_guide=True,
            use_spatial_guide=True,
        )),

    bbox_head=dict(
        type='RetinaDYHead',
        num_classes=10,
        in_channels=512,
        stacked_convs=4,
        feat_channels=512,
        cls_base=-8,
        reg_base=0,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),

    # ==== 训练/测试配置 ====
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            gpu_assign_thr=200),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=500)
)

# ======================================================
# 📦 Dataset 设置
# ======================================================

dataset_type = 'UAVDataset'
data_root = '/root/autodl-tmp/VisDrone_Dataset_COCO_Format/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# ⚡ 优化：去除高开销的 Albumentations 数据增强（可在后期恢复）
albu_train_transforms = [
    dict(type='RandomBrightnessContrast', brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    dict(type='ShiftScaleRotate', shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.3),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
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
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# ⚙️ Dataloader 优化：增大 workers，开启 persistent workers
data = dict(
    samples_per_gpu=6,   # 🚀 从 3 → 6，提升吞吐量
    workers_per_gpu=8,   # 🚀 提高加载速度
    persistent_workers=True,
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
        pipeline=test_pipeline)
)

evaluation = dict(interval=1, metric='bbox')

# ======================================================
# ⚡ 优化器 & 混合精度 AMP
# ======================================================

# ✅ 使用 AmpOptimWrapper 替换旧 optimizer
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='SGD', lr=0.015, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=10, norm_type=2)
)

# ✅ 开启 fp16 混合精度（自动缩放）
fp16 = dict(loss_scale='dynamic')

# ✅ 启用 TF32 和 cuDNN benchmark
# 在运行时添加：
# export NVIDIA_TF32_OVERRIDE=1
# export CUDNN_BENCHMARK=1

# ======================================================
# 🔧 训练控制参数
# ======================================================

find_unused_parameters = True
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[35, 45])
runner = dict(type='EpochBasedRunner', max_epochs=50)
total_epochs = 50

# ✅ 日志配置优化
log_config = dict(
    interval=1000,  # 从200调大，减少I/O占用
    hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
checkpoint_config = dict(interval=5, max_keep_ckpts=3)
