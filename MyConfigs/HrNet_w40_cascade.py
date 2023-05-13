_base_ = './HrNet_w32_cascade.py'
# model settings
model = dict(
    pretrained=None,
    init_cfg=None,
    backbone=dict(
        norm_eval=False,
        init_cfg=None,  
        type='HRNet',
        extra=dict(
            stage2=dict(num_channels=(40, 80)),
            stage3=dict(num_channels=(40, 80, 160)),
            stage4=dict(num_channels=(40, 80, 160, 320))),
        # init_cfg=dict(
        #     type='Pretrained', checkpoint='open-mmlab://msra/hrnetv2_w40')
            ),
    neck=dict(type='HRFPN', in_channels=[40, 80, 160, 320], out_channels=256))

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3,)