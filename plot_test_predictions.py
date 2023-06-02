import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mmcv
from mmengine.utils import track_iter_progress
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector

# Show the results
from mmcv.transforms import LoadImageFromFile, Compose, Resize
import cv2

def draw_bounding_boxes(image, bounding_boxes, scores=None, score_threshold=0.05, backend_args=None, savepath=None):
    # Create figure and axes
    fig, ax = plt.subplots(1, **backend_args)

    # Display the image
    ax.imshow(image)

    # Add bounding boxes to the image
    for i, bbox in enumerate(bounding_boxes):
        if scores is not None and scores[i] < score_threshold:
            continue
        
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min

        # Create a rectangle patch
        rect = patches.Rectangle((x_min, y_min), width, height,
                                 linewidth=2, edgecolor='r', facecolor='none')

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
    plt.close(fig)
    




# define dataloader 
loader = LoadImageFromFile(to_float32=False, color_type='color', imdecode_backend='tifffile', backend_args=None)

# Specify the path to model config and checkpoint file
config_file = '/home/roberto/PythonProjects/S2RAWVessel/checkpoints/vfnet_r18_fpn_1x_vessel/20230518_152156_0.0005/vfnet_r18_fpn_1x_vessel.py'
checkpoint_file = '/home/roberto/PythonProjects/S2RAWVessel/checkpoints/vfnet_r18_fpn_1x_vessel/20230518_152156_0.0005/epoch_239.pth'

# Build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# Init visualizer
visualizer = VISUALIZERS.build(model.cfg.visualizer)
# The dataset_meta is loaded from the checkpoint and
# then pass to the model in init_detector
visualizer.dataset_meta = model.dataset_meta

# test_set
imgs=[{"height":1667,"width":2590,"id":1,"file_name":"day1_g_18_coreg.tif"},{"height":1676,"width":2588,"id":2,"file_name":"day1_g_26_coreg.tif"},{"height":1656,"width":2582,"id":3,"file_name":"day1_g_29_coreg.tif"},{"height":1676,"width":2590,"id":4,"file_name":"day1_g_34_coreg.tif"},{"height":1656,"width":2582,"id":5,"file_name":"day1_g_35_coreg.tif"},{"height":1676,"width":2590,"id":6,"file_name":"day1_g_40_coreg.tif"},{"height":1672,"width":2588,"id":7,"file_name":"day1_g_7_coreg.tif"},{"height":1671,"width":2587,"id":8,"file_name":"day2_g_16_coreg.tif"},{"height":1671,"width":2588,"id":9,"file_name":"day2_g_19_coreg.tif"},{"height":1670,"width":2590,"id":10,"file_name":"day2_g_20_coreg.tif"},{"height":1671,"width":2587,"id":11,"file_name":"day2_g_22_coreg.tif"},{"height":1682,"width":2590,"id":12,"file_name":"day2_g_24_coreg.tif"},{"height":1670,"width":2590,"id":13,"file_name":"day2_g_38_coreg.tif"},{"height":1668,"width":2592,"id":14,"file_name":"day3_g_30_coreg.tif"},{"height":1666,"width":2588,"id":15,"file_name":"day3_g_5_coreg.tif"},{"height":1665,"width":2584,"id":16,"file_name":"day4_g_48_coreg.tif"},{"height":1654,"width":2587,"id":17,"file_name":"day4_g_58_coreg.tif"},{"height":1654,"width":2587,"id":18,"file_name":"day4_g_63_coreg.tif"},{"height":1654,"width":2587,"id":19,"file_name":"day4_g_76_coreg.tif"},{"height":1669,"width":2592,"id":20,"file_name":"day5_g_1_coreg.tif"},{"height":1669,"width":2592,"id":21,"file_name":"day5_g_43_coreg.tif"},{"height":1676,"width":2588,"id":22,"file_name":"day5_g_4_coreg.tif"},{"height":1666,"width":2588,"id":23,"file_name":"day6_g_16_coreg.tif"},{"height":1665,"width":2584,"id":24,"file_name":"day6_g_24_coreg.tif"},{"height":1666,"width":2591,"id":25,"file_name":"day6_g_25_coreg.tif"},{"height":1654,"width":2587,"id":26,"file_name":"day7_g_14_coreg.tif"},{"height":1668,"width":2592,"id":27,"file_name":"day7_g_19_coreg.tif"},{"height":1666,"width":2591,"id":28,"file_name":"day7_g_24_coreg.tif"},{"height":1668,"width":2592,"id":29,"file_name":"day7_g_27_coreg.tif"},{"height":1668,"width":2592,"id":30,"file_name":"day7_g_35_coreg.tif"},{"height":1682,"width":2590,"id":31,"file_name":"day7_g_48_coreg.tif"},{"height":1654,"width":2587,"id":32,"file_name":"day7_g_8_coreg.tif"},{"height":1682,"width":2590,"id":33,"file_name":"day8_g_15_coreg.tif"},{"height":1671,"width":2588,"id":34,"file_name":"day8_g_16_coreg.tif"}]

base_path = '/home/roberto/PythonProjects/S2RAWVessel/mmdetection/data/vessels/imgs/'
outpath_base = '/home/roberto/PythonProjects/S2RAWVessel/output_results/'

file_names = [x['file_name'] for x in imgs]


for idx in range(len(file_names)):
    # Test a single image and show the results
    img_path =base_path + file_names[idx]  # or img = mmcv.imread(img), which will only load it once

    load = loader(results={'img_path': img_path})
    img = load['img']
    result = inference_detector(model, img)
    img = mmcv.imconvert(img, 'bgr', 'rgb')
    print('Inference completed. Saving image...')
    
    predictions = list(result.pred_instances.all_items())
    scores, boxes, labels = predictions
    scores = list(scores[1].detach().cpu().numpy())
    boxes = list(boxes[1].detach().cpu().numpy())

    new_name = file_names[idx].replace('.tif','.png')
    savepath = '/home/roberto/PythonProjects/S2RAWVessel/output_results/'+new_name
    # Draw the bounding boxes on the image
    draw_bounding_boxes(img, boxes, scores = scores, backend_args=dict(figsize=(40, 40), dpi=100), savepath=savepath)
    
        