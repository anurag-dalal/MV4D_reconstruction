import torch
import os
import json
import copy
import numpy as np
from PIL import Image
from random import randint
from tqdm import tqdm
import colorsys
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
        mode='nearest',
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
    # print(frame_diff.shape)
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
        mode='nearest',
        padding_mode='zeros',  # Use 'zeros' padding to handle out-of-bounds
        align_corners=True # align_corners=True is crucial for getting the correct behavior
    )[0] #Return to int and remove batch dimension


    return warped_image

def get_3d_coordinates(
    image: torch.Tensor,
    depth: torch.Tensor,
    cam_params: dict,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Calculates the 3D coordinates of each pixel in an image using depth information
    and camera parameters.  Transforms the coordinates to world space.
    Optionally applies a boolean mask to select which points to process.
    Returns the 3D coordinates and the color values for each pixel.

    Args:
        image: A PyTorch tensor of shape [3, H, W] representing the RGB image.
        depth: A PyTorch tensor of shape [1, H, W] representing the depth map.
        cam_params: A dictionary containing camera parameters, including:
            - 'tanfovx': Tangent of the horizontal field of view.
            - 'tanfovy': Tangent of the vertical field of view.
            - 'viewmatrix': The view matrix (4x4).
            - 'projmatrix': The projection matrix (4x4).  (Not used in this function)
            - 'campos': Camera position (3D coordinates). (Not used directly, related to viewmatrix)
        mask: An optional PyTorch tensor of shape [H, W] representing a boolean mask.
              If provided, only pixels where mask is True will have their 3D
              coordinates and color values calculated.  Pixels where mask is False will be excluded.

    Returns:
        A PyTorch tensor of shape [N, 6] representing the 3D coordinates
        (x, y, z) and color values (r, g, b) of the N valid pixels in world space, where N is the number of True values in the mask.
    """
    # Extract camera parameters
    tanfovx = cam_params.tanfovx
    tanfovy = cam_params.tanfovy
    viewmatrix = cam_params.viewmatrix.squeeze(0)  # Remove batch dimension

    _, height, width = image.shape
    device = image.device

    # Create a meshgrid of pixel coordinates (x, y)
    x = torch.arange(width, device=device).float()
    y = torch.arange(height, device=device).float()
    X, Y = torch.meshgrid(x, y, indexing='xy')  # Use 'xy' indexing

    # Normalize pixel coordinates to the range [-1, 1] (NDC space)
    x_ndc = (2 * X / (width - 1) - 1)
    y_ndc = (2 * Y / (height - 1) - 1)

    # Back-project from NDC to camera space
    x_camera = x_ndc * depth[0, :, :] / tanfovx
    y_camera = y_ndc * depth[0, :, :] / tanfovy
    z_camera = -depth[0, :, :]  # Depth is negative Z in camera space

    # Stack to get 3D coordinates in camera space
    points_camera = torch.stack((x_camera, y_camera, z_camera), dim=-1)  # Shape: [H, W, 3]

    # Convert to homogeneous coordinates
    ones = torch.ones((height, width, 1), device=device)
    points_camera_homogeneous = torch.cat((points_camera, ones), dim=-1)  # Shape: [H, W, 4]

    # Transform to world space using the inverse of the view matrix
    viewmatrix_inv = torch.inverse(viewmatrix)  # Shape: [4, 4]
    points_world_homogeneous = torch.einsum('hwc,cd->hwd', points_camera_homogeneous, viewmatrix_inv)

    # Extract the 3D coordinates (x, y, z) from the homogeneous coordinates
    points_world = points_world_homogeneous[..., :3]  # Shape: [H, W, 3]

    # Extract color values from the image and reshape to [H, W, 3]
    colors = image.permute(1, 2, 0)  # Shape: [H, W, 3]

    # Concatenate 3D coordinates and color values
    points_and_colors = torch.cat((points_world, colors), dim=-1) # Shape: [H, W, 6]

    # Apply the mask if provided
    if mask is not None:
        if mask.shape != (height, width):
            raise ValueError(f"Mask tensor should have shape [H, W], but got {mask.shape}")
        # Use the mask to select only the valid points
        points_and_colors = points_and_colors[mask] # Shape: [N, 6]

    return points_and_colors
def get_camera_matrices(
    image_height: int,
    image_width: int,
    tanfovx: float,
    tanfovy: float,
    viewmatrix: torch.Tensor,
    projmatrix: torch.Tensor,
    campos: torch.Tensor
):
    """
    Calculates the camera-to-world transformation matrix and projection matrix
    from the given camera parameters.  Handles potential issues with input
    tensor shapes.

    Args:
        image_height: Height of the image.
        image_width: Width of the image.
        tanfovx: Tangent of the horizontal field of view.
        tanfovy: Tangent of the vertical field of view.
        viewmatrix: The view matrix (4x4).  Expected shape: (1, 4, 4) or (4, 4).
        projmatrix: The projection matrix (4x4). Expected shape: (1, 4, 4) or (4, 4).
        campos: Camera position (3D coordinates). Expected shape: (3,)

    Returns:
        camera_to_world: A 4x4 PyTorch tensor representing the transformation
            from camera to world coordinates.
        projection_matrix: A 4x4 PyTorch tensor representing the projection matrix.
    """

    # --- Input Validation and Handling ---
    # Ensure viewmatrix and projmatrix are 4x4
    if viewmatrix.ndim == 3:
        if viewmatrix.shape[0] != 1 or viewmatrix.shape[1:] != (4, 4):
            raise ValueError(
                "viewmatrix should have shape (1, 4, 4) or (4, 4), but got"
                f" {viewmatrix.shape}"
            )
        viewmatrix = viewmatrix.squeeze(0)  # Remove the batch dimension
    elif viewmatrix.shape != (4, 4):
        raise ValueError(
            "viewmatrix should have shape (1, 4, 4) or (4, 4), but got"
            f" {viewmatrix.shape}"
        )

    if projmatrix.ndim == 3:
        if projmatrix.shape[0] != 1 or projmatrix.shape[1:] != (4, 4):
            raise ValueError(
                "projmatrix should have shape (1, 4, 4) or (4, 4), but got"
                f" {projmatrix.shape}"
            )
        projmatrix = projmatrix.squeeze(0)  # Remove the batch dimension
    elif projmatrix.shape != (4, 4):
        raise ValueError(
            "projmatrix should have shape (1, 4, 4) or (4, 4), but got"
            f" {projmatrix.shape}"
        )
    if campos.shape != (3,):
        raise ValueError(f"campos should have shape (3,), but got {campos.shape}")

    # --- Calculations ---
    camera_to_world = torch.inverse(viewmatrix)  # Camera-to-world transformation

    projection_matrix = projmatrix

    return camera_to_world, projection_matrix


def get_scale_matrix(sx: float, sy: float, sz: float) -> torch.Tensor:
    """
    Generates a 4x4 scale matrix.

    Args:
        sx: Scaling factor along the x-axis.
        sy: Scaling factor along the y-axis.
        sz: Scaling factor along the z-axis.

    Returns:
        A 4x4 PyTorch tensor representing the scale matrix.
    """
    scale_matrix = torch.eye(4)  # Start with a 4x4 identity matrix
    scale_matrix[0, 0] = sx
    scale_matrix[1, 1] = sy
    scale_matrix[2, 2] = sz
    return scale_matrix



def pixels_to_3d2(curr_data, im, depth, fg_mask):
    """
    Args:
        curr_data: dictionary containing camera settings (GaussianRasterizationSettings)
        im: [3, H, W] image tensor (colors)
        depth: [1, H, W] depth map tensor
        fg_mask: [1, H, W] foreground mask (values >0 are foreground)

    Returns:
        points_3d_rgb: [N, 6] tensor (X, Y, Z, R, G, B) in world coordinates,
                       where N is the number of foreground pixels.
    """
    camera_to_world, projection_matrix = get_camera_matrices(
        curr_data.image_height,
        curr_data.image_width,
        curr_data.tanfovx,
        curr_data.tanfovy,
        curr_data.viewmatrix,
        curr_data.projmatrix,
        curr_data.campos,
    )

    _, height, width = im.shape
    device = im.device

    # Create a meshgrid of pixel coordinates (x, y)
    x = torch.arange(width, device=device).float()
    y = torch.arange(height, device=device).float()
    X, Y = torch.meshgrid(x, y, indexing='xy')

    # Normalize pixel coordinates to NDC
    x_ndc = (2 * X / (width - 1) - 1)
    y_ndc = (2 * Y / (height - 1) - 1)

    # Back-project from NDC to camera space
    x_camera = x_ndc * depth[0, :, :] / curr_data.tanfovx
    y_camera = y_ndc * depth[0, :, :] / curr_data.tanfovy
    z_camera = depth[0, :, :]
    points_camera = torch.stack((x_camera, y_camera, z_camera), dim=-1)  # [H, W, 3]

    # Convert to homogeneous coordinates
    ones = torch.ones((height, width, 1), device=device)
    points_camera_homogeneous = torch.cat((points_camera, ones), dim=-1)  # [H, W, 4]

    # Transform to world space
    points_world_homogeneous = torch.einsum(
        "hwc,cd->hwd", points_camera_homogeneous, camera_to_world
    )
    points_world = points_world_homogeneous[..., :3]  # [H, W, 3]

    # Extract colors
    colors = im.permute(1, 2, 0)  # [H, W, 3]

    # Concatenate points and colors
    points_3d_rgb = torch.cat((points_world, colors), dim=-1)  # [H, W, 6]

    # Apply mask
    fg_mask_bool = fg_mask[0].bool()  # [H, W]
    points_3d_rgb_masked = points_3d_rgb[fg_mask_bool]  # [N, 6]

    return points_3d_rgb_masked
def pixels_to_3d(curr_data, im, depth, fg_mask):
    """
    Args:
        curr_data['cam']: camera settings (GaussianRasterizationSettings)
        im: [3, H, W] image tensor (colors)
        depth: [1, H, W] depth map tensor
        fg_mask: [1, H, W] foreground mask (values >0 are foreground)
        
    Returns:
        points_3d_rgb: [N, 6] tensor (X, Y, Z, R, G, B)
    """

    # Unpack
    H = curr_data['cam'].image_height
    W = curr_data['cam'].image_width
    tanfovx = curr_data['cam'].tanfovx
    tanfovy = curr_data['cam'].tanfovy
    viewmatrix = curr_data['cam'].viewmatrix.squeeze(0)  # [4,4]

    device = im.device

    # Make pixel grid
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )  # [H, W]
    # Backproject to camera space
    # Following standard camera coordinate system convention:
    # X: right, Y: down, Z: forward
    x = x * tanfovx  # X axis points right (unchanged)
    y = y * tanfovy  # Y axis points down (need negative)
    z = torch.ones_like(x)  # Z axis points forward (need negative since depth is positive)

    directions = torch.stack((x, y, z), dim=0)  # [3, H, W]

    # Multiply by depth
    points_camera = directions * depth  # [3, H, W]

    # Reshape for matrix multiply
    points_camera_h = torch.cat([
        points_camera.view(3, -1),
        torch.ones(1, H * W, device=device)
    ], dim=0)  # [4, H*W]

    # Transform to world space
    # points_world_h = (viewmatrix @ points_camera_h)  # [4, H*W]
    # points_world = points_world_h[:3] / points_world_h[3:]  # [3, H*W]
    # scale_matrix = torch.eye(4) * 1
    # scale_matrix = scale_matrix.to(device)
    # points_world = torch.matmul(scale_matrix, points_world_h)[:3]  # [3, H*W]
    campos = curr_data['cam'].campos  # [3]
    campos_h = torch.cat([campos, torch.tensor([1.0], device=campos.device)])  # [4]
    points_world_h = (viewmatrix @ points_camera_h) + campos_h.view(4, 1)  # [4, H*W]
    points_world = points_world_h[:3] # / points_world_h[3:]  # [3, H*W]
    
    # Get corresponding colors
    colors = im.view(3, -1)  # [3, H*W]

    # Apply mask
    mask = (fg_mask > 0).view(-1)  # [H*W]

    points_world = points_world[:, mask]  # [3, N]
    colors = colors[:, mask]              # [3, N]

    # Stack into [N,6]
    points_3d_rgb = torch.cat([points_world, colors], dim=0).transpose(0, 1)  # [N, 6]

    return points_3d_rgb

def remove_low_density_points_in_grid(points, grid_size=0.08, k=27):
    """
    Removes points from a point cloud that fall into grid cells with fewer than k points.

    Args:
        points (torch.Tensor): A tensor of shape [N, 7] representing the point cloud.
            Each point is represented as (X, Y, Z, R, G, B, CamID).
        grid_size (float): The size of the grid cells.
        k (int): The minimum number of points required in a grid cell.

    Returns:
        torch.Tensor: A tensor of shape [M, 7] containing the filtered points,
            where M <= N.
    """
    # Ensure points is a torch tensor
    if not isinstance(points, torch.Tensor):
        raise TypeError("Input 'points' must be a torch.Tensor.")

    # Ensure points has the correct shape
    if points.ndim != 2 or points.shape[1] != 7:
        raise ValueError("Input 'points' must have shape [N, 7].")

    # Ensure grid_size is a float
    if not isinstance(grid_size, (int, float)):
        raise TypeError("Input 'grid_size' must be a float or int.")
    if grid_size <= 0:
        raise ValueError("Input 'grid_size' must be positive.")

    # Ensure k is an integer
    if not isinstance(k, int):
        raise TypeError("Input 'k' must be an integer.")
    if k < 0:
        raise ValueError("Input 'k' must be non-negative.")
    if k == 0:
        return points # return original points if k is zero.

    # Find the min and max coordinates.  Use .min() and .max()
    x_min, y_min, z_min = points[:, 0].min(), points[:, 1].min(), points[:, 2].min()
    x_max, y_max, z_max = points[:, 0].max(), points[:, 1].max(), points[:, 2].max()

    # Calculate the number of grid cells in each dimension.  Use torch.ceil()
    x_bins = int(torch.ceil((x_max - x_min) / grid_size))
    y_bins = int(torch.ceil((y_max - y_min) / grid_size))
    z_bins = int(torch.ceil((z_max - z_min) / grid_size))

    # Calculate the grid indices for each point. Use floor
    x_indices = ((points[:, 0] - x_min) / grid_size).floor().int()
    y_indices = ((points[:, 1] - y_min) / grid_size).floor().int()
    z_indices = ((points[:, 2] - z_min) / grid_size).floor().int()

    # Create a dictionary to store points in each grid cell
    grid_cells = {}
    for i in range(points.shape[0]):
        cell_index = (x_indices[i].item(), y_indices[i].item(), z_indices[i].item())  # Use .item()
        if cell_index not in grid_cells:
            grid_cells[cell_index] = []
        grid_cells[cell_index].append(i)

    # Filter out grid cells with fewer than k points
    dense_cells = [cell_index for cell_index, indices in grid_cells.items() if len(indices) >= k]

    # Collect the indices of the points in the dense cells
    dense_indices = []
    for cell_index in dense_cells:
        dense_indices.extend(grid_cells[cell_index])

    # Use torch.tensor()
    dense_indices_torch = torch.tensor(dense_indices, dtype=torch.long)
    # Use advanced indexing
    filtered_points = points[dense_indices_torch]

    return filtered_points
def pixels_to_3d3(curr_data, im, depth, fg_mask):
    """
    Projects image pixels with depth into 3D world space.

    Args:
        curr_data['cam']: camera settings (GaussianRasterizationSettings)
        im: [3, H, W] image tensor (RGB colors)
        depth: [1, H, W] depth map tensor (Z-depth in camera space)
        fg_mask: [1, H, W] foreground mask (values >0 are foreground)

    Returns:
        points_3d_rgb: [N, 6] tensor (X, Y, Z, R, G, B)
    """
    # Unpack camera parameters
    H = curr_data['cam'].image_height
    W = curr_data['cam'].image_width
    tanfovx = curr_data['cam'].tanfovx
    tanfovy = curr_data['cam'].tanfovy
    viewmatrix = curr_data['cam'].viewmatrix.squeeze(0)  # [4, 4]

    device = im.device

    # Create pixel grid in normalized device coordinates [-1, 1]
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )  # [H, W]

    # Backproject to camera space using intrinsics approximation
    # Camera convention: X (right), Y (down), Z (forward)
    x = -x * tanfovx  # X-right (negate due to convention)
    y = y * tanfovy   # Y-down
    z = torch.ones_like(x)  # Z-forward

    directions = torch.stack((x, y, z), dim=0)  # [3, H, W]

    # Scale directions by depth
    points_camera = directions * depth  # [3, H, W]

    # Homogenize for transformation
    points_camera_h = torch.cat([
        points_camera.view(3, -1),                   # [3, H*W]
        torch.ones(1, H * W, device=device)          # [1, H*W]
    ], dim=0)  # [4, H*W]

    # Invert view matrix to get camera-to-world transform
    viewmatrix_inv = torch.inverse(viewmatrix)  # [4, 4]

    # Transform points to world space
    points_world_h = viewmatrix_inv @ points_camera_h  # [4, H*W]
    points_world = points_world_h[:3] / points_world_h[3:]  # [3, H*W]

    # Get image colors
    colors = im.view(3, -1)  # [3, H*W]

    # Apply foreground mask
    mask = (fg_mask > 0).view(-1)  # [H*W]
    points_world = points_world[:, mask]  # [3, N]
    colors = colors[:, mask]              # [3, N]

    # Combine into [N, 6] (XYZRGB)
    points_3d_rgb = torch.cat([points_world, colors], dim=0).transpose(0, 1)  # [N, 6]

    return points_3d_rgb

def get_changes(prev_params, prev_dataset, curr_dataset):
    rendervar = params2rendervar(prev_params)
    rendervar['means2D'].retain_grad()
    processed_for_flow = process_input_for_flow(prev_dataset, curr_dataset)
    model = ptlflow.get_model("raft", ckpt_path="things")
    model = model.cuda()
    model.eval()
    
    # Initialize a list to collect all points from different frames
    all_points_list = []
    import time
    start = time.time()
    for i in range(len(curr_dataset)):
        curr_data = curr_dataset[i]
        prev_data = prev_dataset[i]
        im, radius, depth = Renderer(raster_settings=prev_data['cam'])(**rendervar)
        seg = curr_data['seg']
        
        # Get optical flow
        with torch.no_grad():
            predictions = model({"images": torch.stack([curr_data['im'], prev_data['im']], dim=0).unsqueeze(0)})
            flow = predictions["flows"][0, 0]
        
        # Apply flow to depth
        prop_depth = apply_flow_torch(depth, flow)
        fg_mask = get_frame_differnce_from_flow(flow)
        
        # print(f"Processing frame {i}...")
        # print(f"im shape: {im.shape}, ({im.dtype})")
        # print(f"seg shape: {seg.shape}, ({seg.dtype})")
        # print(f"depth shape: {depth.shape}, ({depth.dtype})")
        # print(f"flow shape: {flow.shape}, ({flow.dtype})")
        # print(f"prop_depth shape: {prop_depth.shape}, ({prop_depth.dtype})")
        
        # Convert tensors to CPU and numpy for visualization
        # prev_im_np = prev_data['im'].permute(1, 2, 0).detach().cpu().numpy()
        # curr_im_np = curr_data['im'].permute(1, 2, 0).detach().cpu().numpy()
        # prev_depth = depth.squeeze(0).detach().cpu().numpy()
        # curr_depth = prop_depth.squeeze(0).detach().cpu().numpy()
        # fg_mask_np = fg_mask.squeeze(0).detach().cpu().numpy() > 0.5
        
        # curr_depth_corrected = np.where(fg_mask_np, curr_depth, 1)
        # curr_masked_image = np.where(np.expand_dims(fg_mask_np, axis=2), curr_im_np, 0)
        # flow = flow.permute(1, 2, 0).detach().cpu().numpy()
        # flow_viz = flow_utils.flow_to_rgb(flow)  # Represent the flow as RGB colors
        
        # Get 3D points for current frame
        # points = pixels_to_3d3(prev_data, prev_data['im'], depth, fg_mask > -10)
        points = pixels_to_3d(curr_data, curr_data['im'], prop_depth, fg_mask)
        # points = pixels_to_3d(prev_data, prev_data['im'], depth, fg_mask)
        # points = pixels_to_3d2(prev_data['cam'], prev_data['im'], depth, fg_mask > -10)
        
        # Add camera ID as a fourth dimension to help distinguish points from different views
        # Create a tensor with camera ID
        camera_id_tensor = torch.full((points.shape[0], 1), i, dtype=torch.float32, device=points.device)
        
        # Concatenate points and camera ID
        points_with_id = torch.cat([points, camera_id_tensor], dim=1)  # Shape: [N, 7] (X, Y, Z, R, G, B, CamID)
        
        # Add to list of all points
        all_points_list.append(points_with_id)
        
        print(f"Added {points.shape[0]} points from camera {i}")
        # print(curr_dataset[i]['cam'])
        # Clear memory and remove unwanted variables
        del im, radius, depth, seg, flow, prop_depth, fg_mask, points, points_with_id
        torch.cuda.empty_cache()
    # Concatenate all points from different frames
    end = time.time()
    print(f"Time taken to process all frames: {end - start:.2f} seconds")
    if all_points_list:
        all_points = torch.cat(all_points_list, dim=0)
        start = time.time()
        all_points = remove_low_density_points_in_grid(all_points, grid_size=0.05, k=int(len(curr_dataset)*2))
        end = time.time()
        print(f"Time taken to remove low density points: {end - start:.2f} seconds")
        print(f"Total points collected: {all_points.shape[0]}")
        
        # Visualize the combined point cloud
        visualize_point_cloud(all_points, seq=sequence)
    else:
        print("No points collected!")


def visualize_point_cloud(points, seq="unknown"):
    """
    Visualize a point cloud using Open3D.
    
    Args:
        points: Tensor of shape [N, 7] with (X, Y, Z, R, G, B, CamID)
        seq: Sequence name for window title
    """
    import open3d as o3d
    
    # Convert points to numpy for Open3D
    points_np = points.detach().cpu().numpy()
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np[:, :3])  # XYZ coordinates
    
    # Use camera ID to create a colormap for the points
    if points_np.shape[1] >= 7:
        # Option 1: Use the original RGB values
        pcd.colors = o3d.utility.Vector3dVector(points_np[:, 3:6])  # RGB colors
        
        # Create a separate point cloud to show camera grouping with distinct colors
        pcd_cameras = o3d.geometry.PointCloud()
        pcd_cameras.points = o3d.utility.Vector3dVector(points_np[:, :3])
        
        # Create color based on camera ID (last column)
        unique_cameras = np.unique(points_np[:, 6])
        colors = []
        for cam_id in points_np[:, 6]:
            # Assign a distinct color to each camera
            hue = (cam_id / len(unique_cameras)) % 1.0
            # Convert HSV to RGB (simple conversion, assuming S=V=1)
            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            colors.append([r, g, b])
        
        pcd_cameras.colors = o3d.utility.Vector3dVector(np.array(colors))
    else:
        # Just use the RGB colors from the point data
        pcd.colors = o3d.utility.Vector3dVector(points_np[:, 3:6])  # RGB colors
    
    # Create coordinate frame for reference
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    
    # Calculate the point cloud center and scale for better visualization
    center = np.mean(points_np[:, :3], axis=0)
    scale = np.max(np.abs(points_np[:, :3] - center)) * 2
    
    # Visualize the original colored point cloud
    print(f"Visualizing {len(points_np)} points in 3D with original colors")
    o3d.visualization.draw_geometries([pcd, coordinate_frame], 
                                     window_name=f"3D Points (Original Colors) - {seq}",
                                     width=1024, 
                                     height=768,
                                     point_show_normal=False)
    
    # Visualize the point cloud with camera-based coloring            
    if False:                     
        if points_np.shape[1] >= 7:
            print(f"Visualizing {len(points_np)} points in 3D with camera-based colors")
            o3d.visualization.draw_geometries([pcd_cameras, coordinate_frame], 
                                            window_name=f"3D Points (Camera Colors) - {seq}",
                                            width=1024, 
                                            height=768,
                                            point_show_normal=False)

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
    for sequence in ["football"]:
        train(sequence, exp_name)
        torch.cuda.empty_cache()
