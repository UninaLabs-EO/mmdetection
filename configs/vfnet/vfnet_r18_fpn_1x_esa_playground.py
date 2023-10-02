_base_ = [
    '../vfnet/vfnet_r50_fpn_1x_vessel_esa.py',
]




# model settings
model = dict(
    type='VFNet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[0,0,0],
        std=[1,1,1],
        bgr_to_rgb=True,
        pad_size_divisor=8),
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True, # from True
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
        # init_cfg=None),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256, # Up from 128
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=6, # from 5
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='VFNetHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=3,
        feat_channels=256,
        strides=[1, 2, 4, 8, 16],
        center_sampling=False,
        dcn_on_last_conv=False,
        use_atss=True,
        use_vfl=True,
        loss_cls=dict(
            type='VarifocalLoss',
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.5),
        loss_bbox_refine=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=12),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=4,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.25),
        max_per_img=100))