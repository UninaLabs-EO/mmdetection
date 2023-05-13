_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/vessel_detection.py',
    '../_base_/schedules/schedule_Cosine.py', '../_base_/default_runtime.py'
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    )

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001))

# saver_config
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=2),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))
