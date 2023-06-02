from pathlib import Path
import pandas as pd  
import os
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import shutil
import logging



def run_inference(config_file, checkpoint_file, workdir):
    out = os.path.join(workdir, "inference_results.pkl")
    output_dir = os.path.join(workdir, "inference")

    model_name = os.path.splitext(os.path.basename(config_file))[0]

    command = [
        'python', 'tools/test.py', 
        config_file, 
        checkpoint_file,
        '--work-dir', workdir, 
        '--out', out, 
        '--show-dir', output_dir,
        '--custom',
        '--run_name={}_TEST'.format(model_name)
    ]
    subprocess.run(command, check=True)
    
    # delete the output dir if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

def manage_checkpoints(workdir):
    """
    This function manages checkpoint files in a given directory. It reads a JSON file named 'scalars.json',
    finds the row with the maximum 'coco/bbox_mAP_50' value, and deletes all '.pth' files in the directory
    that don't correspond to this maximum value. The function then prints the step and value of the maximum
    'coco/bbox_mAP_50', and the list of remaining '.pth' files.

    Parameters:
    workdir (str): The directory path which contains the checkpoint files and 'scalars.json'.

    Returns:
    None: The function performs operations on files and doesn't return any value.

    Note: 
    This function deletes files in the directory, ensure that you have a backup or you are sure about
    deleting the files before running the function.
    """
    pesi = Path(workdir).glob('*.pth')
    scalars = list(Path(workdir).glob('**/scalars.json'))

    data = pd.read_json(scalars[0], lines=True)
    try:
        row = data[data['coco/bbox_mAP_50'] == data['coco/bbox_mAP_50'].max()]
        step, value = row['step'].values[0], row['coco/bbox_mAP_50'].values[0]
        print(step, value)

        for p in pesi:
            nome = p.name
            if nome != f'epoch_{step}.pth':
                print(f"Deleting {nome}")
                os.remove(p)
    # if the JSON file is empty, print a message
    except:
        print("No mAPs found!")
        for p in pesi:
            print(f"Deleting {nome}")
            os.remove(p)
        
def plot_checkpoints(workdir):
    """
    This function plots the 'coco/bbox_mAP' values in a given directory. It reads a JSON file named 'scalars.json',
    and plots the 'coco/bbox_mAP' values against the 'Epoch' values.

    Parameters:
    workdir (str): The directory path which contains the checkpoint files and 'scalars.json'.

    Returns:
    None: The function plots the graph and doesn't return any value.
    """
    scalars = list(Path(workdir).glob('**/scalars.json'))
    title = list(Path(workdir).glob('**/*.py'))
    title = title[0].stem
    data = pd.read_json(scalars[0], lines=True)
    data = data[data['coco/bbox_mAP_50'].notna()]

    # plot all the coco/bbox_mAP values and save each one in the same folder
    plt.figure(figsize=(5, 5))
    # add style to the plot:
    plt.style.use('seaborn-darkgrid')
    plt.xlabel('Epoch')
    plt.ylabel('mAP@50')
    plt.title(title)
    plt.plot(data['step'], data['coco/bbox_mAP_50'])
    plt.savefig(f'{workdir}/mAP_50.png')
    plt.show()

    # make a plot of the mAP values:
    plt.figure(figsize=(5, 5))
    # add style to the plot:
    plt.style.use('seaborn-darkgrid')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title(title)
    plt.plot(data['step'], data['coco/bbox_mAP'])
    plt.savefig(f'{workdir}/mAP.png')
    plt.show()

    # make a plot of the mAP@75 values:
    plt.figure(figsize=(5, 5))
    # add style to the plot:
    plt.style.use('seaborn-darkgrid')
    plt.xlabel('Epoch')
    plt.ylabel('mAP@75')
    plt.title(title)
    plt.plot(data['step'], data['coco/bbox_mAP_75'])
    plt.savefig(f'{workdir}/mAP_75.png')
    plt.show()  
    
    # make a unique plot of the mAP values:
    plt.figure(figsize=(5, 5))
    # add style to the plot:
    plt.style.use('seaborn-darkgrid')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title(title)
    plt.plot(data['step'], data['coco/bbox_mAP'], label='mAP')
    plt.plot(data['step'], data['coco/bbox_mAP_50'], label='mAP@50')
    plt.plot(data['step'], data['coco/bbox_mAP_75'], label='mAP@75')
    plt.legend()
    plt.savefig(f'{workdir}/mAP_all.png')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', type=str)
    args = parser.parse_args()
    # prepare the logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # create a file handler and save the log in the workdir
    fh = logging.FileHandler(f'{args.workdir}/model_checkpoints.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    
    # Call the function with the directory path
    # manage_checkpoints(args.workdir)
    logger.info("Plotter started!")
    plot_checkpoints(args.workdir)
    logger.info("Plotter finished!")
    # get the config file .py from workdir
    logger.info("Getting config file...")
    config_files = list(Path(args.workdir).glob('**/*.py'))
    # filter files if "config" not in str(config_file):
    config_files = [f for f in config_files if not "config" in str(f)]
    if len(config_files) == 1:
        logger.info("Config file found!")
    else:
        logger.critical(f"More/Less than one config file found! \n Please check the directory: \n {config_files}")
    assert len(config_files) == 1, "More/Less than one config file found"
    config_file = config_files[0]
    logger.info(f"Config file: {config_file}")
    # if config file exists, get the checkpoint file .pth from workdir
    if config_file is not None:
        config_file = config_file.as_posix()
        # get the checkpoint file .pth from workdir
        checkpoint_file = list(Path(args.workdir).glob('*.pth'))[0]
        if checkpoint_file is not None:
            checkpoint_file = str(checkpoint_file)
            # run inference
            print("Running inference...")
            run_inference(config_file, checkpoint_file, args.workdir)
        else:
            print("No checkpoint file found!")

