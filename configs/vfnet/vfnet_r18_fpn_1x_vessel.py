_base_ = [
    '../vfnet/vfnet_r50_fpn_1x_vessel.py',
]

# model
model = dict(
    backbone=dict(
        depth=18,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(in_channels=[64, 128, 256, 512]))


train_dataloader = dict(
    batch_size=2,
    num_workers=8,
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
