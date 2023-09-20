from pathlib import Path
import json
import pandas as pd
import os
import shutil
import argparse

def list_subfolder(dir_path):
    """
    Returns a list of all directories in the given directory path.

    Args:
        dir_path (str): The path to the directory to search for subdirectories.

    Returns:
        List[str]: A list of all subdirectories in the given directory path.
    """
    return [str(path) for path in Path(dir_path).iterdir() if path.is_dir()]


def list_files(dir_path):
    """
    Returns a list of all files in the given directory path and its subdirectories.

    Args:
        dir_path (str): The path to the directory to search for files.

    Returns:
        List[str]: A list of all files in the given directory path and its subdirectories.
    """
    files = []
    for path in Path(dir_path).rglob('*'):
        if path.is_file():
            files.append(str(path))
    return files

def get_files_info(directory):
    """
    Returns a dictionary containing the paths of the config file, scalar file, and weight file.

    Args:
        dir_path (str): The path to the directory to search for files.

    Returns:
        Dict[str, str]: A dictionary containing the paths of the config file, scalar file, and weight file.
    """
    all_files = list_files(directory)

    config_file = [f for f in all_files if 'config.py' in f]
    scalar_file = [f for f in all_files if 'scalars.json' in f]
    weight_file = [f for f in all_files if 'best_coco_bbox' in f or 'epoch' in f]
    
    if not config_file or not scalar_file or not weight_file:
        print(f"Required files not found in training directory: {directory}")
        return None

    return {'config_file': config_file[0], 'scalar_file': scalar_file[0], 'weight_file': weight_file[0]}

def get_best_mAP_inf(scalar_file_path):
    """
    Returns the highest mean average precision (mAP) value for object detection from a given scalar file.

    Args:
        scalar_file_path (str): The path to the scalar file containing the mAP values.

    Returns:
        float: The highest mAP value found in the scalar file.

    Raises:
        FileNotFoundError: If the scalar file cannot be found at the specified path.
    """
    data = pd.read_json(scalar_file_path, lines=True)
    try:
        row = data[data['coco/bbox_mAP_50'] == data['coco/bbox_mAP_50'].max()]
        step, value = row['step'].values[0], row['coco/bbox_mAP_50'].values[0]
        return value
    except:
        print('Not working for the folder: ', scalar_file_path)
        return 0
    

def read_python_file(file_path):
    """
    Reads a Python file and returns its contents as a list of strings.

    Args:
        file_path (str): The path to the Python file to read.

    Returns:
        list of str: The contents of the Python file as a list of strings.

    Raises:
        FileNotFoundError: If the Python file cannot be found at the specified path.
    """
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        return lines
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def find_pattern(lista_righe, pattern):
    """
    Finds the first occurrence of a pattern in a list of strings.

    Args:
        lista_righe (list of str): The list of strings to search for the pattern.
        pattern (str): The pattern to search for in the list of strings.

    Returns:
        str or None: The first string in the list that contains the pattern, or None if the pattern is not found.
    """
    for idx, item in enumerate(lista_righe):
        if pattern in item:
            return item
    return None


if __name__ == '__main__':
    # Example usage:
    # python weights_export.py --weights_dir "/home/roberto/PythonProjects/S2RAWVessel/checkpoints/S2RAW" --output_dir "/home/roberto/PythonProjects/S2RAWVessel/weights_export" --project_name "S2RAW"
    
    # Build parser
    parser = argparse.ArgumentParser(description='Export weights from training directory')
    parser.add_argument('--weights_dir', type=str, help='Path to training models weights directory')
    parser.add_argument('--output_dir', type=str, help='Path to output directory')
    parser.add_argument('--project_name', type=str, help='Name of the project to save the weights')
    args = parser.parse_args()
    
    # Start the main code:
    models_dirs = list_subfolder(args.weights_dir)
    # make a folder for final weights:
    output_weights_folder = args.output_dir + f'/{args.project_name}/'
    print('Exporting into::::>', output_weights_folder)
    os.makedirs(output_weights_folder, exist_ok=True)

    for idx_a in range(len(models_dirs)):
        # mapping the best weights to the training folder
        trainings_dirs = list_subfolder(models_dirs[idx_a])
        print('Model Name: ', Path(models_dirs[idx_a]).stem)
        print(trainings_dirs)
        outfolder = output_weights_folder + '/' + Path(models_dirs[idx_a]).stem
        os.makedirs(outfolder, exist_ok=True)

        best = 0 # init
        global_model_info = []
        
        for idx_b in range(len(trainings_dirs)):
            info = get_files_info(trainings_dirs[idx_b])
            model_hyper_info = {'model_name': Path(models_dirs[idx_a]).stem, 'training_name': Path(trainings_dirs[idx_b]).stem}
            
            if info is not None:
                scalar_file_path = info['scalar_file']
                # read the max mAP value related to training_dirs[idx]
                best_mAP = get_best_mAP_inf(info['scalar_file'])   
                model_hyper_info['best_mAP'] = best_mAP
                # read config.py file and get the hyperparameters:
        
                model_hyper_info['config_file'] = read_python_file(info['config_file'])
                model_hyper_info['config_filepath'] = info['config_file']
                model_hyper_info['scalar_file'] = info['scalar_file']
                model_hyper_info['weight_file'] = info['weight_file']
                global_model_info.append(model_hyper_info)
                
        # loop in the dict, get the max mAP, and copy the corresponding weight, config, and scalar files to the outfolder:
        best = 0
        idx_best = None
        for idx, item in enumerate(global_model_info):
            if item['best_mAP'] > best:
                best = item['best_mAP']
                idx_best = idx
            
        if idx_best is not None:
            print('BEST:', global_model_info[idx_best]['training_name'])
            print('Best config file: ', global_model_info[idx_best]['config_filepath'])
            print(global_model_info[idx_best]['best_mAP'])
            print(global_model_info[idx_best]['model_name'])
            
            best_config = global_model_info[idx_best]['config_filepath']
            best_scalar = global_model_info[idx_best]['scalar_file']
            best_weights = global_model_info[idx_best]['weight_file']    
        
            print(best_weights)
            # copy the best weights, config, and scalar to the outfolder using shutil:
            shutil.copy2(best_weights, outfolder + f'/best_coco_bbox_mAP50_{best:.3}.pth')
            shutil.copy2(best_config, outfolder + '/config.py')
            shutil.copy2(best_scalar, outfolder + '/scalars.json')
            # save the global_model_info to a json file:
            with open(outfolder + '/train_model_info.json', 'w') as outfile:
                json.dump(global_model_info, outfile)
                