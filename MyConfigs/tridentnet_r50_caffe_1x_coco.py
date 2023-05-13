home = "/home/sirbastiano/Documenti/Scripts/MMDETv2/mmdetection/configs/"
_base_ = ["_base_/datasets/wake_detection.py",    #dataset
        "_base_/schedules/schedule_40e.py",    #schedules
        '/_base_/default_runtime.py'
        ]
_base_ = [home+x for x in _base_]
_base_.append("./faster_rcnn_r50_caffe_c4.py",)

model = dict(
    type='TridentFasterRCNN',
    backbone=dict(
        type='TridentResNet',
        trident_dilations=(1, 2, 3),
        num_branch=3,
        test_branch_idx=1,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    roi_head=dict(type='TridentRoIHead', num_branch=3, test_branch_idx=1),
    train_cfg=dict(
        rpn_proposal=dict(max_per_img=500),
        rcnn=dict(
            sampler=dict(num=128, pos_fraction=0.5,
                         add_gt_as_proposals=False))))


data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,)