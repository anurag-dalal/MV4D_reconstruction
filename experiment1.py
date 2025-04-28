import torch
import os
import json
import copy
import numpy as np
from PIL import Image
from random import randint
from tqdm import tqdm
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import setup_camera, l1_loss_v1, l1_loss_v2, weighted_l2_loss_v1, weighted_l2_loss_v2, quat_mult, \
    o3d_knn, params2rendervar, params2cpu, save_params
from external import calc_ssim, calc_psnr, build_rotation, densify, update_params_and_optimizer


import ptlflow
import cv2
from ptlflow.utils import flow_utils

def get_dataset(t, md, seq):
    dataset = []
    for c in range(len(md['fn'][t])):
        w, h, k, w2c = md['w'], md['h'], md['k'][t][c], md['w2c'][t][c]
        cam = setup_camera(w, h, k, w2c, near=1.0, far=100)
        fn = md['fn'][t][c]
        im = np.array(copy.deepcopy(Image.open(f"/home/anurag/Datasets/dynamic/data/{seq}/ims/{fn}")))
        im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255
        seg = np.array(copy.deepcopy(Image.open(f"/home/anurag/Datasets/dynamic/data/{seq}/seg/{fn.replace('.jpg', '.png')}"))).astype(np.float32)
        seg = torch.tensor(seg).float().cuda()
        seg_col = torch.stack((seg, torch.zeros_like(seg), 1 - seg))
        dataset.append({'cam': cam, 'im': im, 'seg': seg_col, 'id': c})
    return dataset


def get_batch(todo_dataset, dataset):
    if not todo_dataset:
        todo_dataset = dataset.copy()
    curr_data = todo_dataset.pop(randint(0, len(todo_dataset) - 1))
    return curr_data


def initialize_params(seq, md):
    init_pt_cld = np.load(f"/home/anurag/Datasets/dynamic/data/{seq}/init_pt_cld.npz")["data"]
    seg = init_pt_cld[:, 6]
    max_cams = 50
    sq_dist, _ = o3d_knn(init_pt_cld[:, :3], 3)
    mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001)
    params = {
        'means3D': init_pt_cld[:, :3],
        'rgb_colors': init_pt_cld[:, 3:6],
        'seg_colors': np.stack((seg, np.zeros_like(seg), 1 - seg), -1),
        'unnorm_rotations': np.tile([1, 0, 0, 0], (seg.shape[0], 1)),
        'logit_opacities': np.zeros((seg.shape[0], 1)),
        'log_scales': np.tile(np.log(np.sqrt(mean3_sq_dist))[..., None], (1, 3)),
        'cam_m': np.zeros((max_cams, 3)),
        'cam_c': np.zeros((max_cams, 3)),
    }
    params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in
              params.items()}
    cam_centers = np.linalg.inv(md['w2c'][0])[:, :3, 3]  # Get scene radius
    scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))
    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'scene_radius': scene_radius,
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float()}
    return params, variables


def initialize_optimizer(params, variables):
    lrs = {
        'means3D': 0.00016 * variables['scene_radius'],
        'rgb_colors': 0.0025,
        'seg_colors': 0.0,
        'unnorm_rotations': 0.001,
        'logit_opacities': 0.05,
        'log_scales': 0.001,
        'cam_m': 1e-4,
        'cam_c': 1e-4,
    }
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def get_loss(params, curr_data, variables, is_initial_timestep):
    losses = {}

    rendervar = params2rendervar(params)
    rendervar['means2D'].retain_grad()
    im, radius, depth = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    curr_id = curr_data['id']
    im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
    losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    segrendervar = params2rendervar(params)
    segrendervar['colors_precomp'] = params['seg_colors']
    seg, _, _, = Renderer(raster_settings=curr_data['cam'])(**segrendervar)
    losses['seg'] = 0.8 * l1_loss_v1(seg, curr_data['seg']) + 0.2 * (1.0 - calc_ssim(seg, curr_data['seg']))

    if not is_initial_timestep:
        is_fg = (params['seg_colors'][:, 0] > 0.5).detach()
        fg_pts = rendervar['means3D'][is_fg]
        fg_rot = rendervar['rotations'][is_fg]

        rel_rot = quat_mult(fg_rot, variables["prev_inv_rot_fg"])
        rot = build_rotation(rel_rot)
        neighbor_pts = fg_pts[variables["neighbor_indices"]]
        curr_offset = neighbor_pts - fg_pts[:, None]
        curr_offset_in_prev_coord = (rot.transpose(2, 1)[:, None] @ curr_offset[:, :, :, None]).squeeze(-1)
        losses['rigid'] = weighted_l2_loss_v2(curr_offset_in_prev_coord, variables["prev_offset"],
                                              variables["neighbor_weight"])

        losses['rot'] = weighted_l2_loss_v2(rel_rot[variables["neighbor_indices"]], rel_rot[:, None],
                                            variables["neighbor_weight"])

        curr_offset_mag = torch.sqrt((curr_offset ** 2).sum(-1) + 1e-20)
        losses['iso'] = weighted_l2_loss_v1(curr_offset_mag, variables["neighbor_dist"], variables["neighbor_weight"])

        losses['floor'] = torch.clamp(fg_pts[:, 1], min=0).mean()

        bg_pts = rendervar['means3D'][~is_fg]
        bg_rot = rendervar['rotations'][~is_fg]
        losses['bg'] = l1_loss_v2(bg_pts, variables["init_bg_pts"]) + l1_loss_v2(bg_rot, variables["init_bg_rot"])

        losses['soft_col_cons'] = l1_loss_v2(params['rgb_colors'], variables["prev_col"])

    loss_weights = {'im': 1.0, 'seg': 3.0, 'rigid': 4.0, 'rot': 4.0, 'iso': 2.0, 'floor': 2.0, 'bg': 20.0,
                    'soft_col_cons': 0.01}
    loss = sum([loss_weights[k] * v for k, v in losses.items()])
    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    return loss, variables


def initialize_per_timestep(params, variables, optimizer):
    pts = params['means3D']
    rot = torch.nn.functional.normalize(params['unnorm_rotations'])
    new_pts = pts + (pts - variables["prev_pts"])
    new_rot = torch.nn.functional.normalize(rot + (rot - variables["prev_rot"]))

    is_fg = params['seg_colors'][:, 0] > 0.5
    prev_inv_rot_fg = rot[is_fg]
    prev_inv_rot_fg[:, 1:] = -1 * prev_inv_rot_fg[:, 1:]
    fg_pts = pts[is_fg]
    prev_offset = fg_pts[variables["neighbor_indices"]] - fg_pts[:, None]
    variables['prev_inv_rot_fg'] = prev_inv_rot_fg.detach()
    variables['prev_offset'] = prev_offset.detach()
    variables["prev_col"] = params['rgb_colors'].detach()
    variables["prev_pts"] = pts.detach()
    variables["prev_rot"] = rot.detach()

    new_params = {'means3D': new_pts, 'unnorm_rotations': new_rot}
    params = update_params_and_optimizer(new_params, params, optimizer)

    return params, variables


def initialize_post_first_timestep(params, variables, optimizer, num_knn=20):
    is_fg = params['seg_colors'][:, 0] > 0.5
    init_fg_pts = params['means3D'][is_fg]
    init_bg_pts = params['means3D'][~is_fg]
    init_bg_rot = torch.nn.functional.normalize(params['unnorm_rotations'][~is_fg])
    neighbor_sq_dist, neighbor_indices = o3d_knn(init_fg_pts.detach().cpu().numpy(), num_knn)
    neighbor_weight = np.exp(-2000 * neighbor_sq_dist)
    neighbor_dist = np.sqrt(neighbor_sq_dist)
    variables["neighbor_indices"] = torch.tensor(neighbor_indices).cuda().long().contiguous()
    variables["neighbor_weight"] = torch.tensor(neighbor_weight).cuda().float().contiguous()
    variables["neighbor_dist"] = torch.tensor(neighbor_dist).cuda().float().contiguous()

    variables["init_bg_pts"] = init_bg_pts.detach()
    variables["init_bg_rot"] = init_bg_rot.detach()
    variables["prev_pts"] = params['means3D'].detach()
    variables["prev_rot"] = torch.nn.functional.normalize(params['unnorm_rotations']).detach()
    params_to_fix = ['logit_opacities', 'log_scales', 'cam_m', 'cam_c']
    for param_group in optimizer.param_groups:
        if param_group["name"] in params_to_fix:
            param_group['lr'] = 0.0
    return variables


def report_progress(params, data, i, progress_bar, every_i=100):
    if i % every_i == 0:
        im, _, _, = Renderer(raster_settings=data['cam'])(**params2rendervar(params))
        curr_id = data['id']
        im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
        psnr = calc_psnr(im, data['im']).mean()
        progress_bar.set_postfix({"train img 0 PSNR": f"{psnr:.{7}f}"})
        progress_bar.update(every_i)


def process_input_for_flow(prev_dataset, curr_dataset):
    combined = []
    for prev, curr in zip(prev_dataset, curr_dataset):
        pair = torch.stack([prev['im'], curr['im']], dim=0)
        combined.append(pair)
    return {"images":torch.stack(combined, dim=0)}
    
import torch.nn.functional as F

def frame_difference_background_removal(background, frame, threshold=3):
    """
    Args:
        background: [C, H, W] tensor (reference background)
        frame: [C, H, W] tensor (current frame)
        threshold: float (threshold to detect moving areas)

    Returns:
        fg_mask: [1, H, W] tensor (binary mask of foreground)
        foreground: [C, H, W] tensor (foreground frame)
    """
    # Convert to grayscale if input is RGB
    if background.shape[0] == 3:
        background_gray = 0.2989 * background[0] + 0.5870 * background[1] + 0.1140 * background[2]
        frame_gray = 0.2989 * frame[0] + 0.5870 * frame[1] + 0.1140 * frame[2]
    else:
        background_gray = background[0]
        frame_gray = frame[0]

    # Apply Gaussian blur (optional: helps reduce noise)
    background_gray = F.avg_pool2d(background_gray.unsqueeze(0).unsqueeze(0), 5, stride=1, padding=2).squeeze(0).squeeze(0)
    frame_gray = F.avg_pool2d(frame_gray.unsqueeze(0).unsqueeze(0), 5, stride=1, padding=2).squeeze(0).squeeze(0)

    # Absolute difference
    diff = frame_gray - background_gray

    # Threshold to create foreground mask
    fg_mask = (diff > (threshold / 255.0)).float()  # threshold normalized to [0, 1]

    # Optionally clean small noise with morphological operations (erosion-dilation)
    fg_mask = F.max_pool2d(fg_mask.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0).squeeze(0)

    # Apply mask to frame
    foreground = frame * fg_mask.unsqueeze(0)

    return fg_mask.unsqueeze(0), foreground

def warp_depth_with_flow(depth, flow, normalized=True):
    """
    Warp a depth image using flow.
    
    Args:
        depth: [1, H, W] tensor
        flow: [2, H, W] tensor
        normalized: bool, whether flow is already normalized to [-1, 1]
        
    Returns:
        warped_depth: [1, H, W] tensor
    """
    h, w = depth.shape[1], depth.shape[2]

    # Create normalized meshgrid [-1, 1]
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, h, device=depth.device),
        torch.linspace(-1, 1, w, device=depth.device),
        indexing='ij'
    )
    base_grid = torch.stack((grid_x, grid_y), dim=2)  # [H, W, 2]

    if not normalized:
        # Flow is in pixels, normalize it
        flow_x = flow[0] / (w / 2)
        flow_y = flow[1] / (h / 2)
    else:
        # Flow is already normalized
        flow_x = flow[0]
        flow_y = flow[1]

    normalized_flow = torch.stack((flow_x, flow_y), dim=2)  # [H, W, 2]

    # Add flow to base grid
    sampling_grid = base_grid + normalized_flow  # [H, W, 2]
    sampling_grid = sampling_grid.unsqueeze(0)   # [1, H, W, 2]

    # Prepare depth for sampling
    depth = depth.unsqueeze(0)  # [1, 1, H, W]

    # Grid sample
    warped_depth = F.grid_sample(
        depth,
        sampling_grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )

    return warped_depth.squeeze(0)  # [1, H, W]

def get_frame_differnce_from_flow(flow, threshold=1.5):
    """
    Convert flow to frame difference.
    
    Args:
        flow: [2, H, W] tensor
        
    Returns:
        frame_diff: [1, H, W] tensor
    """
    # Compute the sum of squares of the flow components
    flow_magnitude_squared = flow[0,:,:]**2 + flow[1,:,:]**2

    # Threshold the flow magnitude
    frame_diff = (flow_magnitude_squared > threshold).float()  # Convert to binary mask
    frame_diff = frame_diff.unsqueeze(0)
    print(frame_diff.shape)
    return frame_diff  # [1, H, W]


def apply_flow_torch(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Applies optical flow to an image using PyTorch tensors.

    Args:
        image: A PyTorch tensor of shape [C, H, W] representing the image, where C is 1 for grayscale
               or 3 for RGB.
        flow: A PyTorch tensor of shape [2, H, W] representing the optical flow.
              flow[0, y, x] is the horizontal (x) displacement at pixel (x, y).
              flow[1, y, x] is the vertical (y) displacement at pixel (x, y).

    Returns:
        A PyTorch tensor of shape [C, H, W] representing the warped image.
        Pixels that flow outside the image boundaries are set to 0.
    """
    # Check input tensor shapes
    if image.ndim != 3 or (image.shape[0] != 3 and image.shape[0] != 1):
        raise ValueError(f"Image tensor should have shape [C, H, W] (C=1 or 3), but got {image.shape}")
    if flow.ndim != 3 or flow.shape[0] != 2 or flow.shape[1:] != image.shape[1:]:
        raise ValueError(f"Flow tensor should have shape [2, H, W], but got {flow.shape}")

    channels, height, width = image.shape
    device = image.device
    # Create a meshgrid of (x, y) coordinates
    x = torch.arange(width, device=device).float()
    y = torch.arange(height, device=device).float()
    # meshgrid returns once the x values for all rows and once the y values for all columns
    # X and Y will contain the x,y coordinates for each pixel
    X, Y = torch.meshgrid(x, y, indexing='xy') # indexing='xy' is crucial for correct flow application
    # Add the flow to the original coordinates to find the source coordinates
    src_x = X + flow[0, :, :]
    src_y = Y + flow[1, :, :]

    # Clip source coordinates to stay within image bounds.  Values outside the range become 0
    src_x = torch.clamp(src_x, 0, width - 1)
    src_y = torch.clamp(src_y, 0, height - 1)

    # Use grid_sample to perform the warping
    # grid_sample requires the input coordinates to be in the range [-1, 1].
    # Normalize the source coordinates to this range.
    src_x_norm = 2 * (src_x / (width - 1)) - 1
    src_y_norm = 2 * (src_y / (height - 1)) - 1

    # Create the grid for grid_sample. grid shape = (1, H, W, 2)
    grid = torch.stack((src_x_norm, src_y_norm), dim=-1).unsqueeze(0)

    # grid_sample performs the warping, using bilinear interpolation.
    warped_image = torch.nn.functional.grid_sample(
        image.unsqueeze(0).float(), # Input needs to be 4D (N, C, H, W).
        grid,
        mode='bilinear',
        padding_mode='zeros',  # Use 'zeros' padding to handle out-of-bounds
        align_corners=True # align_corners=True is crucial for getting the correct behavior
    )[0].int() #Return to int and remove batch dimension


    return warped_image

    
def get_changes(prev_params, prev_dataset, curr_dataset):
    rendervar = params2rendervar(prev_params)
    rendervar['means2D'].retain_grad()
    processed_for_flow = process_input_for_flow(prev_dataset, curr_dataset)
    model = ptlflow.get_model("raft", ckpt_path="things")
    model = model.cuda()
    model.eval()
    for i in range(len(curr_dataset)):
        curr_data = curr_dataset[i]
        prev_data = prev_dataset[i]
        im, radius, depth = Renderer(raster_settings=prev_data['cam'])(**rendervar)
        seg = curr_data['seg']
        # process_input_for_flow(prev_data, curr_data)
        predictions = model({"images": torch.stack([curr_data['im'], prev_data['im']], dim=0).unsqueeze(0)})
        # print('------', predictions["flows"].shape, predictions["flows"].dtype)  # Should be (1, 2, H, W, C) where H and W are the height and width of the images
        flow = predictions["flows"][0, 0]
        # print(depth.shape, flow.shape)
        prop_depth = apply_flow_torch(depth, flow)
        fg_mask = get_frame_differnce_from_flow(flow)
        print(f"im shape: {im.shape}, ({im.dtype})")
        print(f"seg shape: {seg.shape}, ({seg.dtype})")
        print(f"depth shape: {depth.shape}, ({depth.dtype})")
        print(f"flow shape: {flow.shape}, ({flow.dtype})")
        print(f"prop_depth shape: {prop_depth.shape}, ({prop_depth.dtype})")
        import matplotlib.pyplot as plt

        # Convert tensors to CPU and numpy for visualization
        prev_im_np = prev_data['im'].permute(1, 2, 0).detach().cpu().numpy()
        curr_im_np = curr_data['im'].permute(1, 2, 0).detach().cpu().numpy()
        prev_depth = depth.squeeze(0).detach().cpu().numpy()
        curr_depth = prop_depth.squeeze(0).detach().cpu().numpy()
        fg_mask_np = fg_mask.squeeze(0).detach().cpu().numpy() > 0.5
        
        curr_depth_corrected = np.where(fg_mask_np, curr_depth, 1)
        curr_masked_image = np.where(np.expand_dims(fg_mask_np, axis=2), curr_im_np, 0)
        flow = flow.permute(1, 2, 0).detach().cpu().numpy()
        flow_viz = flow_utils.flow_to_rgb(flow)  # Represent the flow as RGB colors
        
        
        
        
        
        # Visualize the images
        fig, axs = plt.subplots(2, 5, figsize=(20, 5))
        
        axs[0][0].imshow(prev_im_np)
        axs[0][0].set_title("prev_im_np")
        axs[0][0].axis("off")
        
        axs[0][1].imshow(prev_depth, cmap="viridis")
        axs[0][1].set_title("prev_depth")
        axs[0][1].axis("off")
        
        axs[1][0].imshow(curr_im_np)
        axs[1][0].set_title("curr_im_np")
        axs[1][0].axis("off")
        
        axs[1][1].imshow(curr_depth_corrected, cmap="viridis")
        axs[1][1].set_title("curr_depth_corrected")
        axs[1][1].axis("off")
        
        axs[1][2].imshow(fg_mask_np, cmap="Greys")
        axs[1][2].set_title("fg_mask_np")
        axs[1][2].axis("off")
        
        axs[1][3].imshow(curr_masked_image)
        axs[1][3].set_title("curr_masked_image")
        axs[1][3].axis("off")
        
        axs[1][4].imshow(flow_viz)
        axs[1][4].set_title("flow_viz")
        axs[1][4].axis("off")
    

        plt.tight_layout()
        plt.show()
        break

    
    
    
    
    
    
def train(seq, exp):
    md = json.load(open(f"/home/anurag/Datasets/dynamic/data/{seq}/train_meta.json", 'r'))  # metadata
    num_timesteps = len(md['fn'])
    
    # Check if we have a saved model to initialize from
    saved_model_path = f"./output/{exp}/{seq}/params.npz"
    if os.path.exists(saved_model_path):
        print(f"Found saved model at {saved_model_path}. Loading for initialization...")
        saved_params = dict(np.load(saved_model_path))
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
        prev_params, variables = initialize_params(seq, md)
        # Update params with values from saved model
        for key in params_from_saved:
            if key in prev_params:
                prev_params[key] = torch.nn.Parameter(params_from_saved[key].contiguous().requires_grad_(True))
    else:
        pass
    # print(len(saved_params['means3D']))
    
    prev_dataset = get_dataset(0, md, seq)
    curr_dataset = get_dataset(1, md, seq)
    get_changes(prev_params, prev_dataset, curr_dataset)
    
    


if __name__ == "__main__":
    exp_name = "exp1"
    for sequence in ["basketball"]:
        train(sequence, exp_name)
        torch.cuda.empty_cache()
