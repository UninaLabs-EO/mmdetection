
_base_ = './RetinaNet_50.py'

model = dict(
    type='RetinaNet',
    backbone=dict(
        in_channels=3,
        with_cp=True,
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101'))
)
work_dir = 'checkpoints/RetinaNet_default'  # Directory to save the model checkpoints and logs for the current experiments.

