_base_ = [
    '../retinanet/retinanet_r50_fpn_1x_vessel.py',
]


# model
model = dict(
    backbone=dict(
        depth=18,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(in_channels=[64, 128, 256, 512]))
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))


train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    )


train_pipeline = [
    dict(type='Mosaic', img_scale=(2048, 2048), pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-1024, -1024)),
    dict(
        type='MixUp',
        img_scale=(2048, 2048),
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', scale=(2048, 2048), keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        color_type='color',
        imdecode_backend='tifffile',
        backend_args=None),
    dict(type='Resize', scale=(2048, 2048), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# TODO: support auto scaling lr
# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
# auto_scale_lr = dict(base_batch_size=16)
