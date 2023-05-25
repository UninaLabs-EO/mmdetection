_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/vessel_detection.py',
    '../_base_/schedules/schedule_520.py', '../_base_/default_runtime.py'
]

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[200, 154, 116], 
        std=[22, 24, 27],
        bgr_to_rgb=True,
        pad_size_divisor=64,
        batch_augments=[dict(type='BatchFixedSizePad', size=(640, 640))]),
    backbone=dict(norm_eval=False,
        depth=18,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(
        in_channels=[64, 128, 256, 512],
        relu_before_extra_convs=True,
        no_norm_on_lateral=True,
        norm_cfg=norm_cfg),
    bbox_head=dict(type='RetinaSepBNHead', num_ins=5, norm_cfg=norm_cfg),
    # training and testing settings
    train_cfg=dict(assigner=dict(neg_iou_thr=0.5)))