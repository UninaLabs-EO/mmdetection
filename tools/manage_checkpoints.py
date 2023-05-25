from pathlib import Path
import pandas as pd  
import os
import argparse
import matplotlib.pyplot as plt

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
    This function plots the 'coco/bbox_mAP_50' values in a given directory. It reads a JSON file named 'scalars.json',
    and plots the 'coco/bbox_mAP_50' values against the 'step' values.

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
    plt.xlabel('step')
    plt.ylabel('mAP@50')
    plt.title(title)
    plt.plot(data['step'], data['coco/bbox_mAP_50'])
    plt.savefig(f'{workdir}/mAP_50.png')
    plt.show()

    # make a plot of the mAP values:
    plt.figure(figsize=(5, 5))
    # add style to the plot:
    plt.style.use('seaborn-darkgrid')
    plt.xlabel('step')
    plt.ylabel('mAP')
    plt.title(title)
    plt.plot(data['step'], data['coco/bbox_mAP'])
    plt.savefig(f'{workdir}/mAP.png')
    plt.show()

    # make a plot of the mAP@75 values:
    plt.figure(figsize=(5, 5))
    # add style to the plot:
    plt.style.use('seaborn-darkgrid')
    plt.xlabel('step')
    plt.ylabel('mAP@75')
    plt.title(title)
    plt.plot(data['step'], data['coco/bbox_mAP_75'])
    plt.savefig(f'{workdir}/mAP_75.png')
    plt.show()  
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', type=str)
    args = parser.parse_args()
    
    # Call the function with the directory path
    # manage_checkpoints(args.workdir)
    plot_checkpoints(args.workdir)

