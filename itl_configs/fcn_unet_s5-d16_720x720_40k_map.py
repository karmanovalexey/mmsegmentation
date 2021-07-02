_base_ = [
    '../configs/_base_/models/fcn_unet_s5-d16.py', '../configs/_base_/datasets/mapillary.py',
    '../configs/_base_/default_runtime.py', '../configs/_base_/schedules/schedule_40k.py'
]

norm_cfg = dict(type='BN', requires_grad=True)

crop_size = (720, 720)
model = dict(
    test_cfg=dict(stride=(1, 1), crop_size=(1072, 1920)),
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

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(pipeline=train_pipeline, data_root='/workspace/Mapillary'),
    val=dict(data_root='/workspace/Mapillary', split='/workspace/mmsegmentation/splits/split.txt'),
    test=dict(data_root='/workspace/Mapillary', split='/workspace/mmsegmentation/splits/split.txt')
)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='WandbLoggerHook', by_epoch=False),
        dict(type='TextLoggerHook', by_epoch=False),
])
gpu_ids = [1]
workflow = [('train', 5), ('val', 1)]