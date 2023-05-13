_base_ = './fovea_r50_fpn_4x4_1x_coco.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))

data = dict(samples_per_gpu=2, workers_per_gpu=2)
