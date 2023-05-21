_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/schedules/schedule_240.py',
    '../_base_/datasets/vessel_detection.py', '../_base_/default_runtime.py'
]

image_size = (2048, 2048)
batch_augments = [dict(type='BatchFixedSizePad', size=image_size)]
norm_cfg = dict(type='BN', requires_grad=True)
checkpoint = 'https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b3_3rdparty_8xb32-aa_in1k_20220119-5b4887a0.pth'  # noqa
model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[200, 154, 116], # [123.675, 116.28, 103.53]
        std=[22, 24, 27],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        batch_augments=None),
    backbone=dict(
        _delete_=True,
        type='EfficientNet',
        arch='b3',
        drop_path_rate=0.2,
        out_indices=(3, 4, 5),
        frozen_stages=0,
        norm_cfg=dict(
            type='SyncBN', requires_grad=True, eps=1e-3, momentum=0.01),
        norm_eval=False,
        init_cfg=dict(
            type='Pretrained', prefix='backbone', checkpoint=checkpoint)),
    neck=dict(
        in_channels=[48, 136, 384],
        start_level=0,
        out_channels=256,
        relu_before_extra_convs=True,
        no_norm_on_lateral=True,
        norm_cfg=norm_cfg),
    bbox_head=dict(type='RetinaSepBNHead', num_ins=5, norm_cfg=norm_cfg),
    # training and testing settings
    train_cfg=dict(assigner=dict(neg_iou_thr=0.5)))

# # dataset settings
# train_pipeline = [
#     dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(
#         type='RandomResize',
#         scale=image_size,
#         ratio_range=(0.8, 1.2),
#         keep_ratio=True),
#     dict(type='RandomCrop', crop_size=image_size),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='PackDetInputs')
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
#     dict(type='Resize', scale=image_size, keep_ratio=True),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(
#         type='PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                    'scale_factor'))
# ]
