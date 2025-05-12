import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import ptlflow
from utils.train_utils import get_dataset

from utils.train_utils import zeroth_initialize_params, zeroth_initialize_optimizer, zeroth_get_loss, zeroth_densify


from utils.metrics import get_metrics


from utils.helpers import get_changes
from utils.voxel_utils import VoxelGridManager
import pandas as pd
import time
from tqdm import tqdm
import open3d as o3d

def train(dataset_path, output_path, exp_name):
    visualize = False
    columns = ['Timestep', 'Num_Iteration', 'Train_L1', 'Train_PSNR', 'Train_SSIM', 'Train_LPIPS', 'Train_MSSSIM', 'Test_L1', 'Test_PSNR', 'Test_SSIM', 'Test_LPIPS', 'Test_MSSSIM', 'Training_Time', 'Num_Params', 'Model_Size']

    df = pd.DataFrame(columns=columns)
    md = json.load(open(f"{dataset_path}/train_meta.json", 'r'))  # metadata
    md_test = json.load(open(f"{dataset_path}/test_meta.json", 'r'))  # metadata
    num_timesteps = len(md['fn'])
    flowmodel = ptlflow.get_model("raft", ckpt_path="things")
    flowmodel = flowmodel.cuda()
    flowmodel.eval()
    
    manager = VoxelGridManager(init_pointcloud_path=dataset_path)
    avg_distance, min_distance, max_distance = manager.get_distances()
    if visualize:
        manager.visualize()
    
    zeroth_params, zeroth_variable = zeroth_initialize_params(dataset_path, md)
    curr_timestep = 0
    zeroth_dataset = get_dataset(curr_timestep, md, dataset_path)
    num_iter_per_timestep = 500
    zeroth_optimizer = zeroth_initialize_optimizer(zeroth_params, zeroth_variable)
    

    start = time.time()
    pbar = tqdm(range(num_iter_per_timestep), desc=f"Timestep {curr_timestep}")
    running_loss = 0.0
    for i in pbar:
        batch_loss = 0.0
        for j in range(len(zeroth_dataset)):
            loss, zeroth_variable = zeroth_get_loss(zeroth_params, zeroth_dataset[j], zeroth_variable)
            batch_loss += loss.item()
            loss.backward()
        running_loss = 0.9 * running_loss + 0.1 * batch_loss if i > 0 else batch_loss
        pbar.set_postfix({"loss": f"{running_loss:.6f}"})
        zeroth_optimizer.step()
        zeroth_optimizer.zero_grad()
        # if i % 10 == 0:
        zeroth_params, zeroth_variable = zeroth_densify(zeroth_params, zeroth_variable, zeroth_optimizer, i, avg_distance * 3, max_iters=num_iter_per_timestep)
    end = time.time()
    Training_Time = end - start
    Train_L1, Train_PSNR, Train_SSIM, Train_LPIPS, Train_MSSSIM, _ = get_metrics(zeroth_params, curr_timestep, md, dataset_path)
    Test_L1, Test_PSNR, Test_SSIM, Test_LPIPS, Test_MSSSIM, _ = get_metrics(zeroth_params, curr_timestep, md_test, dataset_path)
    Num_Params = sum(p.numel() for p in zeroth_params.values() if p.requires_grad)
    Model_Size = Num_Params * 4 / (1024 ** 2)  # in MB
    Timestep = curr_timestep
    Num_Iteration = num_iter_per_timestep
    data = {
        'Timestep': Timestep,
        'Num_Iteration': Num_Iteration,
        'Train_L1': Train_L1,
        'Train_PSNR': Train_PSNR,
        'Train_SSIM': Train_SSIM,
        'Train_LPIPS': Train_LPIPS,
        'Train_MSSSIM': Train_MSSSIM,
        'Test_L1': Test_L1,
        'Test_PSNR': Test_PSNR,
        'Test_SSIM': Test_SSIM,
        'Test_LPIPS': Test_LPIPS,
        'Test_MSSSIM': Test_MSSSIM,
        'Training_Time': Training_Time,
        'Num_Params': Num_Params,
        'Model_Size': Model_Size
    }
    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    df.to_csv(os.path.join(output_path, f"{exp_name}_metrics.csv"), index=False)
    
    manager.reinitialize(zeroth_params['means3D'].detach().cpu().numpy(), zeroth_params['rgb_colors'].detach().cpu().numpy())
    if visualize:   
        manager.visualize()


        
            
    
    
    
    
if __name__ == "__main__":
    exp_name = "start"
    dataset_name = "dynamic"
    dataset_root_path = "/mnt/c/MyFiles/Datasets/dynamic/data"
    output_path = "/home/anurag/codes/MV4D_reconstruction/output"
    
    sequences = ["basketball", "boxes", "football", "juggle", "softball", "tennis"]
    sequences = ["basketball"]
    for sequence in sequences:
        dataset_path = os.path.join(dataset_root_path, sequence)
        output_path = os.path.join(output_path, exp_name, dataset_name, sequence)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        train(dataset_path, output_path, exp_name)