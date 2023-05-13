import torch
from utils import *
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import os
import numpy as np

plt.style.use(['science','no-latex'])

torch.cuda.empty_cache()

os.system('clear')
cwd = os.getcwd()

def test(workdir:str):
     """Executes the test on the selected model.
          Input: selection -> the model selected
                 workdir -> the directory where the weights are stored
          Outputs: saving.pkl and json evals of the model
     """

     config_file_path = [x for x in Path(workdir).glob('**/*.py')]
     assert len(config_file_path) == 1, 'Check configuration files'
     config_path = config_file_path[0]

     weights = getListWeight(workdir)
     for idx, checkpoint in enumerate(weights):
          outfile = workdir +f'/saving_{idx}.pkl'
          eval_type = 'bbox'
          checkpoint = checkpoint.absolute().as_posix()
          print(checkpoint)
          os.system(f'python tools/test.py {config_path} {checkpoint} --work-dir {workdir} --out {outfile} --eval {eval_type}')


def train(selection: str, workdir:str, extra_args=None):
     """Executes the test on the selected model.
          Input: selection -> the model selected
                 wordir -> the directory where the weights are saved
                 extra_args -> argument of training to modify the training
          Outputs: epoch.pth
     """

     def add_extra_args(args:dict):
          if args is None:
               return None
          else:
               stringa = ''     
               for key in args:
                    ####### MAX EPOCHS ########
                    if key == 'max_epochs' and args[key] is not None:
                         MAX_EPOCHS = int(args[key])
                         tmp = f'runner.max_epochs={args[key]}'
                         stringa += tmp + ' '
                    ####### SET LEARNING RATE ########
                    if key == 'lr' and args[key] is not None:
                         tmp = f'optimizer.lr={args[key]}'
                         stringa += tmp + ' '
                    ####### LOAD CHECKPOINT ########
                    if key == 'load_from' and args[key] is not None:
                         tmp = f'load_from={args[key]}'
                         stringa += tmp + ' '
                    ####### WORKDIR ########
                    # if key == 'data_root' and args[key] is not None:
                    #      tmp = f'data.train.ann_file={args[key]}/annotations/instances_train2017.json data.train.img_prefix={args[key]}/train2017/ '
                    #      tmp += f'data.val.ann_file={args[key]}/annotations/instances_val2017.json data.val.img_prefix={args[key]}/val2017/ '
                    #      tmp += f'data.test.ann_file={args[key]}/annotations/instances_val2017.json data.test.img_prefix={args[key]}/val2017/'
                    #      stringa += tmp + ' '
                    ####### WORKDIR ########
                    if key == 'classes' and args[key] is not None:
                         tmp = f'data.train.classes="{args[key]}" '
                         tmp += f'data.val.classes="{args[key]}" '
                         tmp += f'data.test.classes="{args[key]}"'
                         stringa += tmp + ' '
                    ####### LEARNING RATE SCHEDULER ########
                    if key == 'lr_cfg' and args[key] is not None:
                         # Cosinge Annealing Lr Schedule
                         if args[key] == 'CosineAnnealing':
                              tmp = f'lr_config.policy={args[key]} lr_config.min_lr=0'
                              stringa += tmp + ' '
                         elif args[key] == 'step':
                              tmp = f'lr_config.policy={args[key]} lr_config.step="[{(MAX_EPOCHS*2)//3},{(MAX_EPOCHS*8)//9}]"'
                              stringa += tmp + ' '

                    
               print(stringa)
               if stringa != '':
                    return stringa[:-1]
               else:
                    return None


     config_path = selector(selection)
     MAX_TRIES = 1 # max number of training retrials
     count = 0
     while count < MAX_TRIES:
          try:
               stringa = add_extra_args(extra_args)
               if stringa is None:
                    os.system(f'bash tools/dist_train.sh {config_path} 1 --cfg-options work_dir={workdir}')
               else:
                    os.system(f'bash tools/dist_train.sh {config_path} 1 --cfg-options work_dir={workdir} {stringa}')
               print("Execution succeeded.")
               print(f'Retrying... N. of trial:{count}')
               count += 1
               torch.cuda.empty_cache()
          except KeyboardInterrupt:
               print("Execution failed")
               break


def getListWeight(folderPath:str):
     """Return list of weights as PosixPath"""

     def sorter(x):
          # Helper
          x = x.stem.split('_')[-1]
          if x == 'latest':
               return 9999
          else:
               return int(x)
          
     ws = Path(folderPath).glob('*')
     ws = [x for x in ws if x.suffix=='.pth']
     ws.sort(key=sorter)
     return ws


def selector(selection):
     if selection == 'ssd_vgg16':
          config_path = cwd+"/MyConfigs/ssd_vgg.py"
     
     if selection == 'fsaf50':
          config_path = cwd+"/MyConfigs/fsaf_r50_fpn_1x_coco.py"
     
     if selection == 'fcos50':
          config_path = cwd+"/MyConfigs/fcos_r50_caffe_fpn_gn-head_1x_coco.py"

     if selection == 'faster50':
          config_path = cwd+"/MyConfigs/faster_rcnn_r50_fpn.py"
     
     if selection == 'htc50':
          config_path = cwd+"/MyConfigs/htc_without_semantic_r50_fpn_1x_coco.py"

     if selection == 'htc101':
          config_path = cwd+"/MyConfigs/htc_without_semantic_r50_fpn_1x_coco.py"
     
     if selection == 'tridentnet50':
          config_path = cwd+"/MyConfigs/tridentnet_r50_caffe_1x_coco.py"
     
     if selection == 'retina18':
          config_path = cwd+"/MyConfigs/RetinaNet_18.py"
     
     if selection == 'retina34':
          config_path = cwd+"/MyConfigs/RetinaNet_34.py"
     
     if selection == 'retina50':
          config_path = cwd+"/configs/RetinaNet_50.py"
          print("Running config...")

     if selection == 'retina_swin':
          config_path = cwd+"/MyConfigs/retinanet_swin-t-p4-w7_fpn_1x_coco.py"

     if selection == 'retina50_timm':
          config_path = cwd+"/MyConfigs/RetinaNet_r50_timm.py"

     if selection == 'retina101':
          config_path = cwd+"/MyConfigs/RetinaNet_101.py"    

     if selection == 'mask50':
          config_path = cwd+"/MyConfigs/Mask_RCNN_r50.py"

     if selection == 'mask101':
          config_path = cwd+"/MyConfigs/Mask_RCNN_r101.py"

     if selection == 'cascade_mask50':
          config_path = cwd+"/MyConfigs/Cascade_Mask_RCNN.py"
     
     if selection == 'cascade_mask101':
          config_path = cwd+"/MyConfigs/Cascade_Mask_RCNN_101.py"

     if selection == 'detr':
          config_path = cwd+"/MyConfigs/Detr_r50.py"

     if selection == 'centripetalnet':
          config_path = cwd+"/MyConfigs/centripetalnet.py"

     if selection == 'vfnet':
          config_path = cwd+"/MyConfigs/vfnet_r50.py"
     
     if selection == 'efficientdet':
          config_path = cwd+'/MyConfigs/retinanet_effb3_fpn_crop896_8x4_1x_coco.py'
     
     if selection == 'sparce_rcnn50':
          config_path = cwd+"/MyConfigs/sparce_rcnn.py"

     if selection == 'yolov3':
          config_path = cwd+"/MyConfigs/yolov3_d53.py"

     if selection == 'hrnet32':
          config_path = cwd+"/MyConfigs/HrNet_w32_cascade.py"

     if selection == 'hrnet40_cascade':
          config_path = cwd+"/MyConfigs/HrNet_w40_cascade.py"

     if selection == 'fovea50':
          config_path = cwd+"/MyConfigs/fovea_r50_fpn_4x4_1x_coco.py"

     if selection == 'fovea101':
          config_path = cwd+"/MyConfigs/fovea_r101_fpn_4x4_1x_coco.py"

     if selection == 'centernet18':
          config_path = cwd+"/MyConfigs/centernet_resnet18_dcnv2.py"

     return config_path


def plotRes(workdir:str, title:str):
     json_files = Path(workdir).glob('**/*')
     json_files = [x for x in json_files if x.suffix == '.json']
     json_files = [x for x in json_files if x.name.startswith('e')]
     json_files.sort()
     Metrics = {'AP_50':[]}
     for idx in range(len(json_files)):
          try:
               df = pd.read_json(json_files[idx], encoding= 'unicode_escape', lines=True)
               metric = df['metric']
               metrics = metric.to_list()[0]
               AP50 = metrics['bbox_mAP_50']
               Metrics['AP_50'].append(AP50)
          except KeyError:
               Metrics['AP_50'].append(0)

     plt.figure(dpi=300)
     plt.plot(Metrics['AP_50'])
     plt.xlabel('Epoch')
     plt.title(title +' ' + str(max(Metrics['AP_50'])))
     plt.ylabel('$mAP_{50}$')
     plt.savefig(f'{workdir}/{title}.png')
     plt.show(block=False)

def getCurrentTime():
     """Returns the current time as a string """
     now=datetime.now()
     now=now.isoformat()
     now=now.split(".")[0]
     return now


def band_selector(band:str):
     """Returns the current path to band dataset """
     band_dict = {'B2':'data/DATASETS/B2','B3':'data/DATASETS/B3','B4':'data/DATASETS/B4','B8':'data/DATASETS/B8'}
     return band_dict[band]

def plotAll(directory:str):
     folders = Path(directory).glob('**/*')
     folders = [x for x in folders if x.is_dir() and not x.name.startswith('B')]
     print(folders)
     for folder in folders:
          try:
               workdir = folder.as_posix()
               plotRes(workdir, title=folder.name)
          except:
               print(f'Skipping {workdir}')


def keepGoodWeight(directory:str):
     """Function that deletes the weigths base on the max_AP scored."""
     w_paths = Path(directory).glob('**/*')
     w_paths = [x for x in w_paths if x.is_file() and x.name.endswith('.pth')] 
     
     json_paths = Path(directory).glob('**/*')
     json_paths = [y for y in json_paths if y.is_file() and y.name.startswith('eval')] 
     json_paths.sort()

     Metrics = {'AP_50':[]}
     for idx in range(len(json_paths)):
          try:
               df = pd.read_json(json_paths[idx], encoding= 'unicode_escape', lines=True)
               metric = df['metric']
               metrics = metric.to_list()[0]
               AP50 = metrics['bbox_mAP_50']
               Metrics['AP_50'].append(AP50)
          except KeyError:     
               Metrics['AP_50'].append(0)

     
     print(np.argmax(Metrics['AP_50']))
     save = np.argmax(Metrics['AP_50'])
     print(np.max(Metrics['AP_50']))
     print(Metrics['AP_50'])

     # delete weights not necessary
     for w in w_paths:
          if w.stem.endswith(str(np.argmax(Metrics['AP_50'])+1)):
               print(f'found, at {w.name}')
          else:
               os.remove(w)
     
     # delete savings:
     pkl_paths = Path(directory).glob('**/*')
     pkl_paths = [y for y in pkl_paths if y.is_file() and y.name.startswith('saving')]
     for p in pkl_paths:
          os.remove(p)