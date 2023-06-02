from pathlib import Path
import os, subprocess
import shutil

# move to the mmdetection folder
os.chdir("/home/roberto/PythonProjects/S2RAWVessel/mmdetection")

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


if __name__ == "__main__":
    # get a list of all the subfolder in the folder_parent:
    folder_parent = "/home/roberto/PythonProjects/S2RAWVessel/checkpoints"
    folders = [f for f in os.listdir(folder_parent) if os.path.isdir(os.path.join(folder_parent, f))]
    # for each subfolder, if the subfolder startswith a "2" run the inference:
    for folder in folders:
        if folder.startswith("2"):
            try:
                workdir = os.path.join(folder_parent, folder)
                # given the folder, find the checkpoint file (.pth) inside
                checkpoint_file = list(Path(workdir).glob("*.pth"))[0]
                # given the checkpoint file, find the config file (.py) inside
                config_file = list(Path(workdir).glob("*.py"))[0]
            except:
                print("Files not found in folder {}".format(folder))
            finally:
                try:
                    run_inference(config_file, checkpoint_file, workdir)
                except:
                    print('Skip')
                    continue
        else:
            # get a list of all the subfolder in this folder:
            subfolders = [f for f in os.listdir(os.path.join(folder_parent, folder)) if os.path.isdir(os.path.join(folder_parent, folder, f))]
            for subfolder in subfolders:
                try:
                    workdir = os.path.join(folder_parent, folder, subfolder)
                    # given the folder, find the checkpoint file (.pth) inside
                    checkpoint_file = list(Path(workdir).glob("*.pth"))[0]
                    # given the checkpoint file, find the config file (.py) inside
                    config_file = list(Path(workdir).glob("*.py"))[0]
                except:
                    print("Files not found in folder {}".format(subfolder))
                finally:
                    try:
                        run_inference(config_file, checkpoint_file, workdir)
                    except:
                        print('Skip')
                        continue