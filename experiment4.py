import torch
import json
import os
import numpy as np
import ptlflow

from helpers import params2rendervar, o3d_knn, setup_camera, l1_loss_v1, l1_loss_v2, weighted_l2_loss_v1, weighted_l2_loss_v2, quat_mult, \
    o3d_knn, params2rendervar, params2cpu, save_params
from more_helpers import initialize_params, get_dataset, get_changes
from external import calc_ssim, calc_psnr, build_rotation, update_params_and_optimizer,accumulate_mean2d_gradient, remove_points
from diff_gaussian_rasterization import GaussianRasterizer as Renderer


def densify(params, variables, optimizer, clone=True):
    variables = accumulate_mean2d_gradient(variables)
    grads = variables['means2D_gradient_accum'] / variables['denom']
    grads[grads.isnan()] = 0.0
    max_grad = torch.max(grads).item()
    to_delete_grad = 0.7 * max_grad
    grad_mask = grads >= to_delete_grad
    grad_mask = grad_mask.unsqueeze(1)
    opacity_mask = torch.sigmoid(params['logit_opacities']) < 0.7
    large_scale_mask = torch.sum(torch.exp(params['log_scales'])**2, dim=-1) > (variables['scene_radius'] * 0.005 )**2
    large_scale_mask = large_scale_mask.unsqueeze(1)
    print(grad_mask.shape, opacity_mask.shape, large_scale_mask.shape)
    to_delete = torch.logical_or(grad_mask, opacity_mask)
    to_delete = torch.logical_or(to_delete, large_scale_mask)
    # to_delete = to_delete.repeat(1, 3)
    to_delete = to_delete.squeeze(1)
    params, variables = remove_points(to_delete, params, variables, optimizer)
    return params, variables

def initialize_optimizer(params, variables):
    lrs = {
        'means3D': 0.00016 * variables['scene_radius'],
        'rgb_colors': 0.0025,
        'unnorm_rotations': 0.001,
        'logit_opacities': 0.05,
        'log_scales': 0.001,
        'cam_m': 1e-4,
        'cam_c': 1e-4,
    }
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

def get_loss(params, curr_data, data_to_optimize, variables):
    losses = {}

    rendervar = params2rendervar(params)
    rendervar['means2D'].retain_grad()
    im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    curr_id = curr_data['id']
    im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
    losses['im'] = 0.8 * l1_loss_v1(im, data_to_optimize['curr_masked_image']) + 0.2 * (1.0 - calc_ssim(im, data_to_optimize['curr_masked_image']))
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification
    # print(radius.shape, variables['max_2D_radius'].shape)
    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    return losses['im'], variables



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
        prev_params, prev_variables = initialize_params(path, md)
        # Update params with values from saved model
        for key in params_from_saved:
            if key in prev_params:
                prev_params[key] = torch.nn.Parameter(params_from_saved[key].contiguous().requires_grad_(True))
    else:
        print(f"Saved model not found at {previous_model_path}. Aborting...")
        return
    # print(len(saved_params['means3D']))
    print(prev_params['means3D'].shape)
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
        # for k in prev_params.keys():
        #     print(k, prev_params[k].shape)
        
        prev_dataset = get_dataset(i-1, md, dataset_path)
        curr_dataset = get_dataset(i, md, dataset_path)
        all_points, data_to_optimize = get_changes(prev_params, prev_dataset, curr_dataset, flowmodel)
        # all_points = all_points[0:10000, :]
        sq_dist, _ = o3d_knn(all_points[:, :3].cpu().numpy(), 3)
        mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001)
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
        optimizer = initialize_optimizer(new_params, new_variables)
        # new_params = {}
        # for k in prev_params.keys():
        #     new_params[k] = prev_params[k].clone().detach().requires_grad_(True)
        # print(curr_dataset[0]['cam'])
        # for k in prev_params.keys():
            # print(k, prev_params[k].shape, prev_params[k].dtype)
            # print(k, new_params[k].shape, new_params[k].dtype)
            # print(k, type(prev_params[k]), type(new_params[k]))
        rendervar = params2rendervar(new_params)
        im_prev, _, _ = Renderer(raster_settings=curr_dataset[10]['cam'])(**rendervar)
        for i in range(100):
            for j in range(len(curr_dataset)):
                loss, new_variables = get_loss(new_params, curr_dataset[j], data_to_optimize[j], new_variables)
                loss.backward()
            optimizer.step()
            optimizer.zero_grad()    
            if i % 10 == 0:
                # print(grads.shape, torch.max(torch.abs(grads)).item())                
                # print(new_params['means3D'].shape)
                new_paramsparams, new_variables = densify(new_params, new_variables, optimizer)
        rendervar = params2rendervar(new_params)
        # # for k, v in rendervar.items():
        # #     if isinstance(v, torch.Tensor):
        # #         print(f"{k}: {v.shape}, {v.dtype}, {v.device}, requires_grad={v.requires_grad}")
        # #         print(f"    NaNs: {torch.isnan(v).any().item()}, Infs: {torch.isinf(v).any().item()}")
        im_curr, _, _ = Renderer(raster_settings=curr_dataset[10]['cam'])(**rendervar)
        # print(im.shape, im.dtype)
        import matplotlib.pyplot as plt

        # Convert the tensor to a numpy array and transpose it for display
        im_prev = im_prev.clone()
        im_prev = im_prev.to('cpu').detach().permute(1, 2, 0).numpy()
        im_curr = im_curr.clone()
        im_curr = im_curr.to('cpu').detach().permute(1, 2, 0).numpy()        

        # Display the image
        plt.subplot(1, 2, 1)
        plt.imshow(im_prev)
        plt.title('Previous Image')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(im_curr)
        plt.title('Current Image')
        plt.axis('off')
        plt.show()
        break
        
        
        
    
    
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
        
