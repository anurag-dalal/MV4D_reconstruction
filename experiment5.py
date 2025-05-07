import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import ptlflow
from utils.train_utils import initialize_params, o3d_knn, get_dataset, initialize_optimizer, get_loss_initial_timestep, get_loss_other_timestep,densify_initial_timestep, densify_other_timestep, get_loss_other_timestep, initialize_optimizer_other
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
    # columns = ['Timestep', 'Num_Iteration', 'Train_L1', 'Train_PSNR', 'Train_SSIM', 'Train_LPIPS', 'Train_MSSSIM']
    df = pd.DataFrame(columns=columns)
    md = json.load(open(f"{dataset_path}/train_meta.json", 'r'))  # metadata
    md_test = json.load(open(f"{dataset_path}/test_meta.json", 'r'))  # metadata
    num_timesteps = len(md['fn'])
    flowmodel = ptlflow.get_model("raft", ckpt_path="things")
    flowmodel = flowmodel.cuda()
    flowmodel.eval()
    
    manager = VoxelGridManager(init_pointcloud_path=dataset_path)
    if visualize:
        manager.visualize()
    prev_params, prev_variables = initialize_params(dataset_path, md)
    curr_timestep = 0
    prev_dataset = get_dataset(curr_timestep, md, dataset_path)
    num_iter_per_timestep = 300
    optimizer = initialize_optimizer(prev_params, prev_variables)
    
    start = time.time()
    pbar = tqdm(range(num_iter_per_timestep), desc=f"Timestep {curr_timestep}")
    running_loss = 0.0
    for i in pbar:
        batch_loss = 0.0
        for j in range(len(prev_dataset)):
            loss, prev_variables = get_loss_initial_timestep(prev_params, prev_dataset[j], prev_variables)
            batch_loss += loss.item()
            loss.backward()
        running_loss = 0.9 * running_loss + 0.1 * batch_loss if i > 0 else batch_loss
        pbar.set_postfix({"loss": f"{running_loss:.6f}"})
        optimizer.step()
        optimizer.zero_grad()
        if i % 10 == 0:
            prev_params, prev_variables = densify_initial_timestep(prev_params, prev_variables, optimizer, i)
    end = time.time()
    Training_Time = end - start
    Train_L1, Train_PSNR, Train_SSIM, Train_LPIPS, Train_MSSSIM, _ = get_metrics(prev_params, curr_timestep, md, dataset_path)
    Test_L1, Test_PSNR, Test_SSIM, Test_LPIPS, Test_MSSSIM, _ = get_metrics(prev_params, curr_timestep, md_test, dataset_path)
    Num_Params = sum(p.numel() for p in prev_params.values() if p.requires_grad)
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
    
    manager.reinitialize(prev_params['means3D'].detach().cpu().numpy(), prev_params['rgb_colors'].detach().cpu().numpy())
    if visualize:   
        manager.visualize()
    
            
    
    for i in range(1, num_timesteps - 1):
        curr_timestep = i
        prev_dataset = get_dataset(curr_timestep-1, md, dataset_path)
        curr_dataset = get_dataset(curr_timestep, md, dataset_path)
        all_points, remove_points, data_to_optimize = get_changes(prev_params, prev_dataset, curr_dataset, flowmodel, manager.voxel_size)
        sq_dist, _ = o3d_knn(all_points[:, :3].cpu().numpy(), 3)
        mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001)
        new_numpoints = all_points.shape[0]
        new_params = {
            'means3D': torch.nn.Parameter(all_points[:,0:3].cuda().float().contiguous().requires_grad_(True)),
            'rgb_colors': torch.nn.Parameter(all_points[:,3:6].cuda().float().contiguous().requires_grad_(True)),
            # 'seg_colors': torch.nn.Parameter(all_points[:,3:6].cuda().float().contiguous().requires_grad_(True)),
            'unnorm_rotations': torch.nn.Parameter(
                                    torch.tensor([1.0, 0.0, 0.0, 0.0], device='cuda')
                                    .repeat(new_numpoints, 1)
                                    .float()
                                    .contiguous()
                                    .requires_grad_(True)
                                ),
            'logit_opacities': torch.nn.Parameter(torch.ones(new_numpoints,1).cuda().float().contiguous().requires_grad_(True)),
            'log_scales': torch.nn.Parameter((torch.tensor(np.tile(np.log(np.sqrt(mean3_sq_dist))[..., None], (1, 3)))).cuda().float().contiguous().requires_grad_(True)),
            'cam_m': torch.nn.Parameter(prev_params['cam_m'].clone().cuda().float().contiguous().requires_grad_(True)),
            'cam_c': torch.nn.Parameter(prev_params['cam_c'].clone().cuda().float().contiguous().requires_grad_(True))
        }
        new_variables = {'max_2D_radius': torch.zeros(new_params['means3D'].shape[0]).cuda().float(),
                 'scene_radius': torch.tensor(prev_variables['scene_radius']).clone().cuda().float(),
                 'means2D_gradient_accum': torch.zeros(new_params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(new_params['means3D'].shape[0]).cuda().float()}
        optimizer = initialize_optimizer_other(new_params, new_variables)
        
        num_iter_per_timestep = 100
        start = time.time()
        pbar = tqdm(range(num_iter_per_timestep), desc=f"Timestep {curr_timestep} pre refinement")
        running_loss = 0.0
        for j in pbar:
            batch_loss = 0.0
            for j in range(len(curr_dataset)):
                loss, new_variables = get_loss_other_timestep(new_params, curr_dataset[j], data_to_optimize[j], new_variables)
                batch_loss += loss.item()
                loss.backward()
            running_loss = 0.9 * running_loss + 0.1 * batch_loss if j > 0 else batch_loss
            pbar.set_postfix({"loss": f"{running_loss:.6f}"})
            optimizer.step()
            optimizer.zero_grad()    
            if i % 20 == 0:
                new_params, new_variables = densify_other_timestep(new_params, new_variables, optimizer)
        
        indices_to_delete = manager.check_if_included_in_voxel(remove_points[:,:3])
        
        # Create filtered versions of prev_params by removing points using indices_to_delete
        filtered_prev_params = {}
        for k, v in prev_params.items():
            if k in ['means3D', 'rgb_colors', 'unnorm_rotations', 'logit_opacities', 'log_scales']:
                filtered_prev_params[k] = prev_params[k][~indices_to_delete]
            else:
                filtered_prev_params[k] = prev_params[k]
        
        # Now combine new_params with filtered_prev_params
        for k, v in new_params.items():
            if k in ['means3D', 'rgb_colors', 'unnorm_rotations', 'logit_opacities', 'log_scales']:
                new_params[k] = torch.nn.Parameter(torch.cat((new_params[k], filtered_prev_params[k]), dim=0).float().contiguous().requires_grad_(True))
            # Cam parameters are already copied and don't need concatenation
        new_variables = {'max_2D_radius': torch.zeros(new_params['means3D'].shape[0]).cuda().float(),
                 'scene_radius': torch.tensor(prev_variables['scene_radius']).clone().cuda().float(),
                 'means2D_gradient_accum': torch.zeros(new_params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(new_params['means3D'].shape[0]).cuda().float()}
        
        optimizer = initialize_optimizer(new_params, new_variables)
        pbar = tqdm(range(num_iter_per_timestep), desc=f"Timestep {curr_timestep} post refinement")
        running_loss = 0.0
        for j in pbar:
            batch_loss = 0.0
            for j in range(len(curr_dataset)):
                loss, new_variables = get_loss_initial_timestep(new_params, curr_dataset[j], new_variables)
                batch_loss += loss.item()
                loss.backward()
            running_loss = 0.9 * running_loss + 0.1 * batch_loss if j > 0 else batch_loss
            pbar.set_postfix({"loss": f"{running_loss:.6f}"})
            optimizer.step()
            optimizer.zero_grad()    
            if i % 10 == 0:
                new_params, new_variables = densify_initial_timestep(new_params, new_variables, optimizer)
                new_params, new_variables = densify_other_timestep(new_params, new_variables, optimizer) 
        
        end = time.time()
        Training_Time = end - start
        Train_L1, Train_PSNR, Train_SSIM, Train_LPIPS, Train_MSSSIM, _ = get_metrics(new_params, curr_timestep, md, dataset_path)
        Test_L1, Test_PSNR, Test_SSIM, Test_LPIPS, Test_MSSSIM, _ = get_metrics(new_params, curr_timestep, md_test, dataset_path)
        Num_Params = sum(p.numel() for p in new_params.values() if p.requires_grad)
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
        prev_params = new_params
        prev_variables = new_variables
            
    
    
    
    
if __name__ == "__main__":
    exp_name = "start"
    dataset_name = "dynamic"
    dataset_root_path = "/home/anurag/Datasets/dynamic/data"
    output_path = "/home/anurag/Codes/MV4D_reconstruction/output"
    
    sequences = ["basketball", "boxes", "football", "juggle", "softball", "tennis"]
    sequences = ["basketball"]
    for sequence in sequences:
        dataset_path = os.path.join(dataset_root_path, sequence)
        output_path = os.path.join(output_path, exp_name, dataset_name, sequence)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        train(dataset_path, output_path, exp_name)