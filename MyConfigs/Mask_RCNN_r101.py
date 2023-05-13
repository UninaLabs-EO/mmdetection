
_base_ = './Mask_RCNN_r50.py'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))


work_dir = 'checkpoints/Mask_RCNN_r101_FPN_default'  # Directory to save the model checkpoints and logs for the current experiments.

