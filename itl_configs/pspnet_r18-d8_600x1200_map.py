_base_ = [
    '../configs/_base_/models/pspnet_r50-d8.py',
    '../configs/_base_/datasets/mapillary.py',
    '../configs/_base_/default_runtime.py',
    './_base_/schedules/schedule_500k_adamw.py'
]

model = dict(
    pretrained='torchvision://resnet18',
    backbone=dict(type='ResNet', depth=18),
    decode_head=dict(
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(in_channels=256, channels=64))


# model settings
norm_cfg = dict(type='BN', requires_grad=True)
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0004, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))
optimizer_config = dict()

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

crop_size = (600, 1200)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

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
    samples_per_gpu=12,
    workers_per_gpu=12,
    train=dict(pipeline=train_pipeline, data_root='/workspace/Mapillary'),
    val=dict(data_root='/workspace/Mapillary', split='/workspace/mmsegmentation/splits/demo_split.txt'),
    test=dict(data_root='/workspace/Mapillary', split='/workspace/mmsegmentation/splits/demo_split.txt')
)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='WandbLoggerHook', by_epoch=False),
        dict(type='TextLoggerHook', by_epoch=False),
])
evaluation = dict(interval=16000, metric='mIoU')
