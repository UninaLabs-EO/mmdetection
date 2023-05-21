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
