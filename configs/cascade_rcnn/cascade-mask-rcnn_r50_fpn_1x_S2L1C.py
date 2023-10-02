_base_ = [
    '../_base_/models/cascade-mask-rcnn_r50_fpn.py',
    '../_base_/datasets/esa_vessel_segmentation.py',
    '../_base_/schedules/schedule_120.py', '../_base_/esa_runtime.py'
]

