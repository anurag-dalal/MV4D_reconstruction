import torch
import ptlflow
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import params2rendervar
import open3d as o3d
import numpy as np
from helpers import o3d_knn, setup_camera
import copy
import colorsys
from PIL import Image

# This function is used to process the input data for the flow model.
# It takes two datasets (previous and current) and combines the images from both datasets into a single tensor.
# The output is a dictionary with a single key "images" that contains the combined images.
def process_input_for_flow(prev_dataset, curr_dataset):
    combined = []
    for prev, curr in zip(prev_dataset, curr_dataset):
        pair = torch.stack([prev['im'], curr['im']], dim=0)
        combined.append(pair)
    return {"images":torch.stack(combined, dim=0)}

# This function is used to convert the flow tensor to a frame difference tensor.
# It computes the sum of squares of the flow components and applies a threshold to create a binary mask.
# The output is a tensor with shape [1, H, W] where H and W are the height and width of the input flow tensor.
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


# This function calculates the reverse optical flow mathematically without using the flow model again
# It takes the forward flow as input and returns an approximation of the reverse flow
# The function uses the forward flow to warp the coordinates and calculate the reverse flow
def calculate_reverse_flow(forward_flow):
    """
    Calculate reverse optical flow mathematically without using the flow model.
    
    Args:
        forward_flow: [2, H, W] tensor representing flow from frame A to frame B
        
    Returns:
        reverse_flow: [2, H, W] tensor representing flow from frame B to frame A
    """
    # Get the dimensions of the flow
    _, height, width = forward_flow.shape
    device = forward_flow.device
    
    # Create a meshgrid of coordinates
    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, device=device).float(),
        torch.arange(width, device=device).float(),
        indexing='ij'
    )
    
    # Calculate the target coordinates by adding the flow
    target_x = x_coords + forward_flow[0]
    target_y = y_coords + forward_flow[1]
    
    # Create a meshgrid for the target frame
    y_target, x_target = torch.meshgrid(
        torch.arange(height, device=device).float(),
        torch.arange(width, device=device).float(),
        indexing='ij'
    )
    
    # Initialize the reverse flow
    reverse_flow = torch.zeros_like(forward_flow)
    
    # Round the target coordinates to the nearest pixel
    target_x_rounded = torch.round(target_x).long().clamp(0, width - 1)
    target_y_rounded = torch.round(target_y).long().clamp(0, height - 1)
    
    # For each target pixel, calculate the reverse flow
    for y in range(height):
        for x in range(width):
            tx = target_x_rounded[y, x]
            ty = target_y_rounded[y, x]
            
            # Set the reverse flow at the target position
            # The reverse flow is the negative of the offset to the source
            reverse_flow[0, ty, tx] = x - tx
            reverse_flow[1, ty, tx] = y - ty
    
    # Smooth the reverse flow to fill holes (simple averaging)
    kernel_size = 3
    padding = kernel_size // 2
    avg_kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device) / (kernel_size * kernel_size)
    
    # Apply smoothing separately to each channel
    reverse_flow_smoothed_x = torch.nn.functional.conv2d(
        reverse_flow[0].unsqueeze(0).unsqueeze(0), 
        avg_kernel, 
        padding=padding
    ).squeeze()
    
    reverse_flow_smoothed_y = torch.nn.functional.conv2d(
        reverse_flow[1].unsqueeze(0).unsqueeze(0), 
        avg_kernel, 
        padding=padding
    ).squeeze()
    
    # Combine the smoothed channels
    reverse_flow_smoothed = torch.stack([reverse_flow_smoothed_x, reverse_flow_smoothed_y], dim=0)
    
    return reverse_flow_smoothed


# This function applies optical flow to an image using PyTorch tensors.
# It takes an image tensor and a flow tensor as input and returns the warped image.
# The function uses bilinear interpolation to warp the image based on the flow.
# The output is a tensor of the same shape as the input image.
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

# This function converts pixel coordinates to 3D points in world space.
# It takes the camera settings, image tensor, depth map, and foreground mask as input.
# The output is a tensor of shape [N, 6] where N is the number of points.
# Each point is represented as (X, Y, Z, R, G, B).
# The function uses the camera settings to backproject the pixel coordinates into 3D space.
# The depth map is used to scale the pixel coordinates to 3D points.
# The foreground mask is used to filter out points that are not in the foreground.
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

# This function removes points from a point cloud that fall into grid cells with fewer than k points.
# It takes a tensor of points and a grid size as input.
# The function first calculates the grid indices for each point based on its coordinates.
# It then creates a dictionary to store points in each grid cell.
# The grid cells with fewer than k points are filtered out.
# The output is a tensor of points that are in dense grid cells.
# The function uses torch operations for efficient computation.
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
    if points.ndim != 2 or not (points.shape[1] == 7 or points.shape[1] == 6):
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

# This function collects points from the previous and current datasets.
# It uses the Gaussian rasterizer to render the points and applies optical flow to propagate depth.
# The function also filters out low-density points using a grid-based approach.
# need downsmpling of points, cause too many points to render gpu out of memory
# strategy 1 select only n number of cameras
def get_changes(prev_params, prev_dataset, curr_dataset, flowmodel, grid_size=0.08):
    
    cameras_to_select = [0, 1, 2, 4, 8, 12, 16, 20, 24]
    rendervar = params2rendervar(prev_params)
    rendervar['means2D'].retain_grad()
    processed_for_flow = process_input_for_flow(prev_dataset, curr_dataset)
    model = ptlflow.get_model("raft", ckpt_path="things")
    flowmodel = flowmodel.cuda()
    flowmodel.eval()
    to_return = []
    # Initialize a list to collect all points from different frames
    all_points_list = []
    remove_from_grid_list = []
    for i in range(len(curr_dataset)):
    # for i in cameras_to_select:
        curr_data = curr_dataset[i]
        prev_data = prev_dataset[i]
        
        
        # Get optical flow
        with torch.no_grad():
            im, radius, depth = Renderer(raster_settings=prev_data['cam'])(**rendervar)
            predictions = flowmodel({"images": torch.stack([curr_data['im'], prev_data['im']], dim=0).unsqueeze(0)})
            flow = predictions["flows"][0, 0]
            predictions = flowmodel({"images": torch.stack([prev_data['im'], curr_data['im']], dim=0).unsqueeze(0)})
            # Calculate reverse flow mathematically instead of running the flow model again
            reverse_flow = predictions["flows"][0, 0]
        
        # Apply flow to depth
        prop_depth = apply_flow_torch(depth, flow)
        fg_mask = get_frame_differnce_from_flow(flow)
        reversed_fg_mask = get_frame_differnce_from_flow(reverse_flow)   
        points = pixels_to_3d(curr_data, curr_data['im'], prop_depth, fg_mask)
        points_to_remove = pixels_to_3d(prev_data, prev_data['im'], depth, reversed_fg_mask)
        
        # Create masked version of current image by applying foreground mask
        # Equivalent to: curr_masked_image = np.where(np.expand_dims(fg_mask_np, axis=2), curr_im_np, 0)
        # For torch: We need to expand fg_mask to match image dimensions and use it for masking
        curr_masked_image = curr_data['im'] * fg_mask  # Broadcasting will expand fg_mask across the channel dimension
        
        # Add camera ID as a fourth dimension to help distinguish points from different views
        # Create a tensor with camera ID
        camera_id_tensor = torch.full((points.shape[0], 1), i, dtype=torch.float32, device=points.device)
        
        # Concatenate points and camera ID
        points_with_id = torch.cat([points, camera_id_tensor], dim=1)  # Shape: [N, 7] (X, Y, Z, R, G, B, CamID)
        
        # Add to list of all points
        all_points_list.append(points_with_id)
        remove_from_grid_list.append(points_to_remove)
        # print(flow.shape, reverse_flow.shape)
        # print(f"Added {points.shape[0]} points from camera {i}")
        to_return.append({
            "prev_rendered": im,
            "prev_depth": depth,
            "flow": flow,
            "reverse_flow": reverse_flow,  # Add the reverse flow to the returned dictionary
            "prop_depth": prop_depth,
            "fg_mask": fg_mask,
            "reversed_fg_mask": reversed_fg_mask,
            "curr_masked_image": curr_masked_image  # Add the masked image to the dictionary
        })
        # print(curr_dataset[i]['cam'])
        # Clear memory and remove unwanted variables
        del im, radius, depth, flow, reverse_flow, prop_depth, fg_mask, points, points_with_id, curr_masked_image
        torch.cuda.empty_cache()
    # only keep the points from the selected cameras
    all_points_list = [all_points_list[i] for i in cameras_to_select]
    remove_from_grid_list = [remove_from_grid_list[i] for i in cameras_to_select]
    # Concatenate all points from different frames
    if all_points_list:
        all_points = torch.cat(all_points_list, dim=0)
        remove_points = torch.cat(remove_from_grid_list, dim=0)
        all_points = remove_low_density_points_in_grid(all_points, grid_size=grid_size, k=int(len(curr_dataset)*2))
        remove_points = remove_low_density_points_in_grid(remove_points, grid_size=grid_size, k=int(len(curr_dataset)*2))
        # print(f"Total points collected: {all_points.shape[0]}")        
        # Visualize the combined point cloud
        # visualize_point_cloud(all_points, seq='none')
    else:
        print("No points collected!")
    return all_points, remove_points, to_return    
    
        
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
            
def initialize_params(path, md):
    init_pt_cld = np.load(path)["data"]
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

def get_dataset(t, md, path):
    dataset = []
    for c in range(len(md['fn'][t])):
        w, h, k, w2c = md['w'], md['h'], md['k'][t][c], md['w2c'][t][c]
        cam = setup_camera(w, h, k, w2c, near=1.0, far=100)
        fn = md['fn'][t][c]
        im = np.array(copy.deepcopy(Image.open(f"{path}/ims/{fn}")))
        im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255
        seg = np.array(copy.deepcopy(Image.open(f"{path}/seg/{fn.replace('.jpg', '.png')}"))).astype(np.float32)
        seg = torch.tensor(seg).float().cuda()
        seg_col = torch.stack((seg, torch.zeros_like(seg), 1 - seg))
        dataset.append({'cam': cam, 'im': im, 'seg': seg_col, 'id': c})
    return dataset