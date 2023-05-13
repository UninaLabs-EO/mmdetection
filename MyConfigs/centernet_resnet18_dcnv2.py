home = "/home/sirbastiano/Documenti/Scripts/MMDETv2/mmdetection/configs/"

_base_ = ["_base_/datasets/wake_detection.py",    #dataset
        "_base_/schedules/schedule_40e.py",    #schedules
        '/_base_/default_runtime.py'
        ]

_base_ = [home+x for x in _base_]


model = dict(
    type='CenterNet',
    backbone=dict(
        type='ResNet',
        depth=18,
        norm_eval=False,
        norm_cfg=dict(type='BN'),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(
        type='CTResNetNeck',
        in_channel=512,
        num_deconv_filters=(256, 128, 64),
        num_deconv_kernels=(4, 4, 4),
        use_dcn=True),
    bbox_head=dict(
        type='CenterNetHead',
        num_classes=1,
        in_channel=64,
        feat_channel=64,
        loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
        loss_wh=dict(type='L1Loss', loss_weight=0.1),
        loss_offset=dict(type='L1Loss', loss_weight=1.0)),
    train_cfg=None,
    test_cfg=dict(topk=100, local_maximum_kernel=3, max_per_img=100))


data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='CocoDataset',
        classes=('wake', ),
        ann_file='data/coco/annotations/instances_train2017.json',
        img_prefix='data/coco/train2017/',
        pipeline=[
            dict(type='LoadImageFromFile', color_type='color'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=[(768, 768)], keep_ratio=False),
            dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
            dict(type='RandomFlip', flip_ratio=0.5, direction='vertical'),
            dict(type='RandomFlip', flip_ratio=0.5, direction='diagonal'),
            dict(
                type='Normalize',
                mean=[128, 128, 128],
                std=[70, 70, 70],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='CocoDataset',
        classes=('wake', ),
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/',
        pipeline=[
            dict(type='LoadImageFromFile', color_type='color'),
            dict(type='Resize', img_scale=[(768, 768)], keep_ratio=False),
            dict(
                type='Normalize',
                mean=[128, 128, 128],
                std=[70, 70, 70],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(
                type='Collect',
                meta_keys=('filename', 'ori_shape', 'img_shape',
                           'scale_factor','border'),
                keys=['img']),
            dict(type='WrapFieldsToLists')
        ]),
    test=dict(
        type='CocoDataset',
        classes=('wake', ),
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/',
        pipeline=[
            dict(type='LoadImageFromFile', color_type='color'),
            dict(type='Resize', img_scale=[(768, 768)], keep_ratio=False),
            dict(
                type='Normalize',
                mean=[128, 128, 128],
                std=[70, 70, 70],
                to_rgb=True),
            dict(
                type='RandomCenterCropPad',
                ratios=None,
                border=None,
                mean=[128, 128, 128],
                std=[70, 70, 70],
                to_rgb=True,
                test_mode=True,
                test_pad_mode=['logical_or', 31],
                test_pad_add_pix=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(
                type='Collect',
                meta_keys=('filename', 'ori_shape', 'img_shape',
                           'scale_factor','border'),
                keys=['img']),
            dict(type='WrapFieldsToLists')
        ]))