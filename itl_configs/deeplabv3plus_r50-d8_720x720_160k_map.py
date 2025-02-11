_base_ = [
    '../configs/_base_/models/deeplabv3plus_r50-d8.py', '../configs/_base_/datasets/mapillary.py',
    '../configs/_base_/default_runtime.py', './_base_/schedules/schedule_160k_adamw.py'
]

norm_cfg = dict(type='BN', requires_grad=True)

crop_size = (720, 720)
model = dict(
    test_cfg=dict(mode='whole'),
    backbone=dict(norm_cfg=norm_cfg),
    decode_head=dict(num_classes=66, norm_cfg=norm_cfg),
    auxiliary_head=dict(num_classes=66, norm_cfg=norm_cfg))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1920, 1080), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001)

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3,
    train=dict(pipeline=train_pipeline, data_root='/workspace/Mapillary'),
    val=dict(data_root='/workspace/Mapillary', split='/workspace/mmsegmentation/splits/split.txt'),
    test=dict(data_root='/workspace/Mapillary', split='/workspace/mmsegmentation/splits/split.txt')
)
resume_from = '/workspace/mmsegmentation/work_dirs/deeplabv3plus_r50-d8_720x720_160k_map/iter_160000.pth'
log_config = dict(
    interval=50,
    hooks=[
        dict(type='WandbLoggerHook', by_epoch=False),
        dict(type='TextLoggerHook', by_epoch=False),
])
