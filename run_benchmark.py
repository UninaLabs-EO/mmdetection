from pathlib import Path
import os, subprocess
import shutil

# move to the mmdetection folder
os.chdir("/home/roberto/PythonProjects/S2RAWVessel/mmdetection")

TASK = "inference"
REPEATER = "1"
output_dir = "/home/roberto/PythonProjects/S2RAWVessel/output_results/BENCHMARK"


def run_benchmark(config_file, checkpoint_file, workdir):
    out = os.path.join(workdir, "inference_results.pkl")
    output_dir = os.path.join(workdir, "inference")

    model_name = os.path.splitext(os.path.basename(config_file))[0]

    command = [
        'python', 'tools/analysis_tools/benchmark.py', 
        config_file, 
        '--checkpoint', checkpoint_file,
        '--task', TASK,
        '--max-iter', '150',
        '--dataset-type', 'train',
        '--repeat-num', REPEATER, 
        '--work-dir', output_dir,
    ]
    subprocess.run(command, check=True)
    
    
    
if __name__ == '__main__':
    workdir = '/home/roberto/PythonProjects/S2RAWVessel/checkpoints/rtmdet_l_8xb32-300e_vessel/20230529_172534_0.005'
    # get the config file and the checkpoint file in the workdir
    config_file = list(Path(workdir).glob("*.py"))[0]
    checkpoint_file = list(Path(workdir).glob("*.pth"))[0]
    
    assert config_file.exists(), f"Config file {config_file} does not exist"
    assert checkpoint_file.exists(), f"Checkpoint file {checkpoint_file} does not exist"
    
    # create a subfolder in output_dir with the name of the model
    model_name = os.path.splitext(os.path.basename(config_file))[0]
    output_subdir = os.path.join(output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    run_benchmark(config_file, checkpoint_file, output_subdir)
    