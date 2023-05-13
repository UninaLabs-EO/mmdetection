from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from pathlib import Path
import numpy as np
import mmcv
import matplotlib.pyplot as plt
import cv2
import os
import fiftyone as fo
from fiftyone import ViewField as F
global LABEL, MODEL_NAME

def model_load(model_cp, config):
     model = init_detector(config, model_cp, device='cuda:0')
     return model

def dataset_parser(IMAGES_DIR,JSON_PATH,LABEL):
     # Load COCO formatted dataset
     dataset = fo.Dataset.from_dir(
     dataset_type=fo.types.COCODetectionDataset,
     data_path=IMAGES_DIR,
     labels_path=JSON_PATH,
     include_id=True,
     label_field='GT',)
     # Verify that the class list for our dataset was imported
     # print(coco_dataset.default_classes)  # ['airplane', 'apple', ...]
     print(dataset)
     return dataset

# MODEL
CFG = 'checkpoints/Craters/cascade_mask50/cascade_mask50_2022-08-05T15:15:10_768_25e_Craters_lr_0.002_CosineAnnealing/Cascade_Mask_RCNN.py'
CHP = 'checkpoints/Craters/cascade_mask50/cascade_mask50_2022-08-05T15:15:10_768_25e_Craters_lr_0.002_CosineAnnealing/epoch_10.pth'
MODEL_NAME = 'Cascade_Mask_RCNN'
model = model_load(model_cp=CHP, config=CFG)
# DATASET
IMAGES_DIR='data/DATASETS/Craters/val2017'
JSON_PATH='data/DATASETS/Craters/annotations/instances_val2017.json'
LABEL='crater'
dataset = dataset_parser(IMAGES_DIR,JSON_PATH,LABEL)

for sample in dataset:
     # sample = fo.Sample(filepath="data/DATASETS/Craters/val2017/539.jpg")
    # img = cv2.imread("data/DATASETS/Craters/val2017/539.jpg")
    img = cv2.imread(sample.filepath)
    h,w,c=img.shape
    result = inference_detector(model, img)
    boxes, masks = result
    X = boxes[0]

    sample[MODEL_NAME] = fo.Detections(
        detections=[
            fo.Detection(label=LABEL, confidence=X[4], bounding_box=[X[0]/w, X[1]/h, X[2]/w-X[0]/w, X[3]/h-X[1]/h]) for X in boxes[0]
        ]
    )
    sample.save()


session = fo.launch_app(dataset)

# Only contains detections with confidence >= 0.75
high_conf_view = dataset.filter_labels(MODEL_NAME, F("confidence") > 0.75)

# Evaluate the predictions in the `faster_rcnn` field of our `high_conf_view`
# with respect to the objects in the `ground_truth` field
results = high_conf_view.evaluate_detections(
    MODEL_NAME,
    gt_field="GT_detections",
    eval_key="eval",
    compute_mAP=True,
)

# Get the 10 most common classes in the dataset
counts = dataset.count_values("GT_detections.detections.label")
classes_top10 = sorted(counts, key=counts.get, reverse=True)[:10]

# Print a classification report for the top-10 classes
with open(f"{MODEL_NAME}_PR.txt", 'w') as f: 
     results.print_report(classes=classes_top10)
     f.write(results.print_report(classes=classes_top10))


print(results.mAP())

plot = results.plot_pr_curves(classes=[LABEL])
plot.show()

