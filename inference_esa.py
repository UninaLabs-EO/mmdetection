import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mmcv
from mmengine.utils import track_iter_progress
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector
import json 
from pathlib import Path
import numpy as np


# Show the results
from mmcv.transforms import LoadImageFromFile, Compose, Resize
import cv2

def draw_bounding_boxes(image, bounding_boxes, scores=None, score_threshold=0.05, backend_args=None, savepath=None, gt_boxes=None):
    # Create figure and axes
    fig, ax = plt.subplots(1, **backend_args)

    # Display the image
    ax.imshow(image)

    # Add gt_boxes to the image
    if gt_boxes is not None:
        for i, bbox in enumerate(gt_boxes):
            x_min, y_min, width, height = bbox

            # Create a rectangle patch
            rect = patches.Rectangle((x_min, y_min), width, height,
                                    linewidth=2, edgecolor='w', facecolor='none')

            # Add the rectangle to the axes
            ax.add_patch(rect)
            ax.axis(False)

    # Add bounding boxes to the image
    for i, bbox in enumerate(bounding_boxes):
        if scores is not None and scores[i] < score_threshold:
            continue
        
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min

        # Create a rectangle patch
        rect = patches.Rectangle((x_min, y_min), width, height,
                                 linewidth=2, edgecolor='y', facecolor='none')

        # Add the rectangle to the axes
        ax.add_patch(rect)
        ax.axis(False)
        # Add the score as text
        if scores is not None:
            score = scores[i]
            ax.text(x_max+5, y_max+5, f'Score: {score:.2f}',
                    color='white', fontsize=8, bbox=dict(facecolor='r', alpha=0.7))
    
    if savepath is not None:
        fig.savefig(savepath)
    # Show the image with bounding boxes
    plt.show()
    plt.close()


def get_annotations_from_coco_json(coco_json_file, image_filename):
    """Get annotations from a COCO JSON file for a given image.

    Args:
        coco_json_file (str): Path to COCO JSON file.
        image_filename (str): Filename of image.

    Returns:
        annotations (list): List of annotations for the given image.
        gt_boxes (list): List of ground truth bounding boxes for the given image.
    """
    annotations = []
    try:
        with open(coco_json_file, 'r') as f:
            coco_data = json.load(f)
    except FileNotFoundError:
        print(f"Could not find file {coco_json_file}")
        return annotations
    except json.JSONDecodeError:
        print(f"Could not decode JSON from {coco_json_file}")
        return annotations

    # Extract image ID corresponding to the given image filename
    image_id = None
    for img in coco_data.get('images', []):
        if img.get('file_name') == image_filename:
            image_id = img.get('id')
            break

    if image_id is None:
        print(f"No matching image found for filename {image_filename}")
        return annotations

    # Extract annotations for the image
    for annotation in coco_data.get('annotations', []):
        if annotation.get('image_id') == image_id:
            annotations.append(annotation)

    gt_boxes = [x['bbox'] for x in annotations]
    
    return annotations, gt_boxes

# define dataloader 
loader = LoadImageFromFile(to_float32=False, color_type='color', imdecode_backend='tifffile', backend_args=None)

# Specify the path to model config and checkpoint file
config_file = '/home/roberto/PythonProjects/S2RAWVessel/checkpoints/S2L1C/vfnet_r101_fpn_1x_esa/20230919_232958_LR_0.0005_BATCH_4_IMG_2816_MEAN_[46,53,51]_STD_[30,34,42]/vfnet_r101_fpn_1x_esa.py'
foldPath = Path(config_file).parent
# list all file sin folderpath:
chkpts = list(foldPath.glob('*.pth'))
checkpoint_file = [x for x in chkpts if 'best' in x.name][0].as_posix()
print(checkpoint_file)
print('Loading checkpoint: ', checkpoint_file)
# Build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

data_root = '/home/roberto/PythonProjects/S2RAWVessel/mmdetection/data/S2ESA/'
model.cfg.test_dataloader = dict(
                    batch_size=1,
                    num_workers=2,
                    persistent_workers=True,
                    drop_last=False,
                    sampler=dict(type='DefaultSampler', shuffle=False),
                    dataset=dict(
                        type='CocoDataset',
                        data_root=data_root,
                        metainfo=dict(classes=('Vessel', ), palette=[(220, 20, 60)]),
                        ann_file='annotations/test.json',
                        data_prefix=dict(img='imgs/'),
                        test_mode=True,
                        filter_cfg=dict(filter_empty_gt=True),
                        pipeline=[
                            dict(
                                type='LoadImageFromFile',
                                to_float32=True,
                                color_type='color',
                                imdecode_backend='tiffile',
                                backend_args=None),
                            dict(type='Resize', scale=(2816, 2816), keep_ratio=True),
                            dict(type='LoadAnnotations', with_bbox=True),
                            dict(
                                type='PackDetInputs',
                                meta_keys=('img', 'img_id', 'img_path', 'img_shape', 'ori_shape', 'scale', 'scale_factor', 'keep_ratio', 'homography_matrix', 'gt_bboxes', 'gt_ignore_flags', 'gt_bboxes_labels'))
                        ],
                        backend_args=None))


# Init visualizer
visualizer = VISUALIZERS.build(model.cfg.visualizer)
# The dataset_meta is loaded from the checkpoint and
# then pass to the model in init_detector
visualizer.dataset_meta = model.dataset_meta

# READ JSON FILE TEST:
json_file_path = data_root + '/annotations/test.json'
json_file = json.load(open(json_file_path))
imgs = json_file['images']

base_path = data_root + '/imgs/'
outpath_base = '/home/roberto/PythonProjects/S2RAWVessel/output_results/S2L1C/'

file_names = [x['file_name'] for x in imgs]

# take 4 random indexes between 0 and len(filenames)
idxs = np.random.choice(len(file_names), size=3, replace=False)
print('Selected indexes:', idxs)
for idx in idxs:
    # Test a single image and show the results
    img_path =base_path + file_names[idx]  # or img = mmcv.imread(img), which will only load it once
    annot, gt_boxes = get_annotations_from_coco_json(json_file_path, file_names[idx])

    load = loader(results={'img_path': img_path})
    img = load['img']
    result = inference_detector(model, img)
    img = mmcv.imconvert(img, 'bgr', 'rgb')
    print('Inference completed. Saving image...')
    
    predictions = list(result.pred_instances.all_items())
    
    keyholder={}
    for item in predictions:
        keyholder[item[0]]=item[1]
        
    scores, boxes, labels = keyholder['scores'], keyholder['bboxes'], keyholder['labels']
    scores = list(scores.detach().cpu().numpy())
    boxes = list(boxes.detach().cpu().numpy())

    new_name = file_names[idx].replace('.tiff','.png')
    savepath = '/home/roberto/PythonProjects/S2RAWVessel/output_results/S2L1C/'+ new_name
    # Draw the bounding boxes on the image
    draw_bounding_boxes(img, boxes, scores = scores, backend_args=dict(figsize=(20, 20), dpi=100), savepath=savepath, score_threshold=0.5, gt_boxes=gt_boxes)
    
