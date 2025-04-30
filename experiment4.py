import torch
import json
import os
import numpy as np
import ptlflow


from more_helpers import initialize_params, get_dataset, get_changes



def train(dataset_path, previous_model_path, exp_name):
    md = json.load(open(f"{dataset_path}/train_meta.json", 'r'))  # metadata
    num_timesteps = len(md['fn'])
    flowmodel = ptlflow.get_model("raft", ckpt_path="things")
    flowmodel = flowmodel.cuda()
    flowmodel.eval()
    
    if os.path.exists(previous_model_path):
        print(f"Found saved model at {previous_model_path}. Loading for initialization...")
        saved_params = dict(np.load(previous_model_path))
        
        
        # Get the initial timestep data from the saved model
        params_from_saved = {
            'means3D': torch.tensor(saved_params['means3D'][0]).cuda().float(),
            'rgb_colors': torch.tensor(saved_params['rgb_colors'][0]).cuda().float(),
            'seg_colors': torch.tensor(saved_params['seg_colors']).cuda().float(),
            'unnorm_rotations': torch.tensor(saved_params['unnorm_rotations'][0]).cuda().float(),
            'logit_opacities': torch.tensor(saved_params['logit_opacities']).cuda().float(),
            'log_scales': torch.tensor(saved_params['log_scales']).cuda().float(),
        }
        
        # Initialize parameters with saved model data
        path = f"{dataset_path}/init_pt_cld.npz"
        prev_params, variables = initialize_params(path, md)
        # Update params with values from saved model
        for key in params_from_saved:
            if key in prev_params:
                prev_params[key] = torch.nn.Parameter(params_from_saved[key].contiguous().requires_grad_(True))
    else:
        print(f"Saved model not found at {previous_model_path}. Aborting...")
        return
    # print(len(saved_params['means3D']))
    
    # for i in range(1,num_timesteps):
    for i in range(1,3):    
        # Get the prev timestep data from the saved model
        params_from_saved = {
            'means3D': torch.tensor(saved_params['means3D'][i-1]).cuda().float(),
            'rgb_colors': torch.tensor(saved_params['rgb_colors'][i-1]).cuda().float(),
            'seg_colors': torch.tensor(saved_params['seg_colors']).cuda().float(),
            'unnorm_rotations': torch.tensor(saved_params['unnorm_rotations'][i-1]).cuda().float(),
            'logit_opacities': torch.tensor(saved_params['logit_opacities']).cuda().float(),
            'log_scales': torch.tensor(saved_params['log_scales']).cuda().float(),
        }
        
        for key in params_from_saved:
            if key in prev_params:
                prev_params[key] = torch.nn.Parameter(params_from_saved[key].contiguous().requires_grad_(False))
        for k in prev_params.keys():
            print(k, prev_params[k].shape)
        
        prev_dataset = get_dataset(i-1, md, dataset_path)
        curr_dataset = get_dataset(i, md, dataset_path)
        all_points, data_to_optimize = get_changes(prev_params, prev_dataset, curr_dataset, flowmodel)
        print(all_points.shape)
        
        # print(f"Optimizing for {dataset_path} timestep {i}...")
        
        # new_params = {
        #     'means3D': torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)),
        #     'rgb_colors': init_pt_cld[:, 3:6],
        #     'seg_colors': np.stack((seg, np.zeros_like(seg), 1 - seg), -1),
        #     'unnorm_rotations': np.tile([1, 0, 0, 0], (seg.shape[0], 1)),
        #     'logit_opacities': np.zeros((seg.shape[0], 1)),
        #     'log_scales': np.tile(np.log(np.sqrt(mean3_sq_dist))[..., None], (1, 3)),
        #     'cam_m': np.zeros((max_cams, 3)),
        #     'cam_c': np.zeros((max_cams, 3)),
        # }
        new_numpoints = all_points.shape[0]
        new_params = {
            'means3D': torch.nn.Parameter(all_points[:,0:3].cuda().float().contiguous().requires_grad_(True)),
            'rgb_colors': torch.nn.Parameter(all_points[:,3:6].cuda().float().contiguous().requires_grad_(True)),
            'unnorm_rotations': torch.nn.Parameter(
                                    torch.tensor([1.0, 0.0, 0.0, 0.0], device='cuda')
                                    .repeat(new_numpoints, 1)
                                    .float()
                                    .contiguous()
                                    .requires_grad_(True)
                                ),
            'logit_opacities': torch.nn.Parameter(torch.ones(new_numpoints,1).cuda().float().contiguous().requires_grad_(True)),
            'log_scales': torch.nn.Parameter(torch.ones(new_numpoints,3).cuda().float().contiguous().requires_grad_(True))}
        
        for k in new_params.keys():
            print(k, new_params[k].shape)
        for i in range(100):
            loss, variables = get_loss(params, curr_data, variables, is_initial_timestep)
            loss.backward()
        
        
        
    
    
if __name__ == "__main__":
    exp_name = "exp2"
    dataset_name = "dynamic"
    daaset_root_path = "/home/anurag/Datasets/dynamic/data"
    output_path = "/home/anurag/Codes/MV4D_reconstruction/output"
    
    sequences = ["basketball", "boxes", "football", "juggle", "softball", "tennis"]
    sequences = ["basketball"]
    for sequence in sequences:
        dataset_path = os.path.join(daaset_root_path, sequence)
        output_path = os.path.join(output_path, exp_name, dataset_name, sequence)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        previous_model_path = f"/home/anurag/Codes/MV4D_reconstruction/output/exp1/{sequence}/params.npz"
        train(dataset_path, previous_model_path, exp_name)
        torch.cuda.empty_cache()
