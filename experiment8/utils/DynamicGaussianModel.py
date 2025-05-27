import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nerfstudio.field_components.encodings import SHEncoding
from diff_gaussian_rasterization_with_depth import GaussianRasterizer as Renderer
from pytorch_msssim import SSIM
from math import exp
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
import ptlflow
import copy

class GaussianModel():

    def o3d_knn(self, pts, num_knn):
        indices = []
        sq_dists = []
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        for p in pcd.points:
            [_, i, d] = pcd_tree.search_knn_vector_3d(p, num_knn + 1)
            indices.append(i[1:])
            sq_dists.append(d[1:])
        return np.array(sq_dists), np.array(indices)
    
    def __init__(self, point_cloud_data, dyanmic_dataset_manager, max_cams=None, cam_m=None, cam_c=None):
        if isinstance(point_cloud_data, torch.Tensor):
            point_cloud_data = point_cloud_data.cpu().numpy()
        sq_dist, _ = self.o3d_knn(point_cloud_data[:, :3], 3)
        mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001)

        if cam_m is None or cam_c is None:
            cam_m = np.zeros((max_cams, 3))
            cam_c = np.zeros((max_cams, 3))

        self.params = {
            'means3D': point_cloud_data[:, :3],
            'rgb_colors': point_cloud_data[:, 3:6],
            'unnorm_rotations': np.tile([1, 0, 0, 0], (point_cloud_data.shape[0], 1)),
            'logit_opacities': np.ones((point_cloud_data.shape[0], 1)),
            'log_scales': np.tile(np.log(np.sqrt(mean3_sq_dist))[..., None], (1, 3)),
            'cam_m': cam_m,
            'cam_c': cam_c,
        }

        self.params = {
            k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
            for k, v in self.params.items()
        }

        self.cam_centers = np.linalg.inv(dyanmic_dataset_manager.train_md['w2c'][0])[:, :3, 3]
        self.scene_radius = 1.1 * np.max(
            np.linalg.norm(self.cam_centers - np.mean(self.cam_centers, 0)[None], axis=-1)
        )
        self.variables = {
            'max_2D_radius': torch.zeros(self.params['means3D'].shape[0]).cuda().float(),
            'scene_radius': self.scene_radius,
            'means2D_gradient_accum': torch.zeros(self.params['means3D'].shape[0]).cuda().float(),
            'denom': torch.zeros(self.params['means3D'].shape[0]).cuda().float()
        }

    def params2rendervar(self):
        rendervar = {
            'means3D': self.params['means3D'],
            'colors_precomp': self.params['rgb_colors'],
            'rotations': torch.nn.functional.normalize(self.params['unnorm_rotations']),
            'opacities': torch.sigmoid(self.params['logit_opacities']),
            'scales': torch.exp(self.params['log_scales']),
            'means2D': torch.zeros_like(self.params['means3D'], requires_grad=True, device="cuda") + 0
        }
        return rendervar
    
    def render(self, cam):
        rendervar = self.params2rendervar()
        # rendervar['means2D'].retain_grad()
        im, radius, depth = Renderer(raster_settings=cam)(**rendervar)
        return im, radius, depth
    
    def save(self, path):
        torch.save(self.params, path)

    def load(self, dyanmic_dataset_manager, path):
        self.params = torch.load(path)
        self.params = {k: torch.nn.Parameter(v).cuda().float().contiguous().requires_grad_(True) for k, v in
              self.params.items()}
        self.cam_centers = np.linalg.inv(dyanmic_dataset_manager.train_md['w2c'][0])[:, :3, 3]
        self.scene_radius = 1.1 * np.max(np.linalg.norm(self.cam_centers - np.mean(self.cam_centers, 0)[None], axis=-1))
        self.variables = {'max_2D_radius': torch.zeros(self.params['means3D'].shape[0]).cuda().float(),
                 'scene_radius': self.scene_radius,
                 'means2D_gradient_accum': torch.zeros(self.params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(self.params['means3D'].shape[0]).cuda().float()}


class VoxelModelManager():
    def __init__(self, voxel_size, origin=None):
        self.voxel_size = voxel_size
        self.points = None
        self.origin = origin

    def initialize_points(self, points):
        # points in Nx3 format
        if isinstance(points, torch.Tensor):
            self.points = points.detach().cpu().numpy()
        elif isinstance(points, GaussianModel):
            self.points = points.params['means3D'].detach().cpu().numpy()

        self.origin = self.points.mean(axis=0)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(self.points)
        self.voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size=self.voxel_size)
        # Store the origin of the voxel grid for proper alignment
        voxels = self.voxel_grid.get_voxels()
        if voxels and self.origin is None:
            min_indices = np.array([voxel.grid_index for voxel in voxels]).min(axis=0)
            self.origin = min_indices * self.voxel_size
    
    def indices_of_points_to_remove(self, points_to_remove):
        if isinstance(points_to_remove, torch.Tensor):
            points_to_remove = points_to_remove.detach().cpu().numpy()
        elif isinstance(points_to_remove, GaussianModel):
            points_to_remove = points_to_remove.params['means3D'].detach().cpu().numpy()

        # Create point cloud for self.points (main points)
        # pcd_main = o3d.geometry.PointCloud()
        # pcd_main.points = o3d.utility.Vector3dVector(self.points)
        # pcd_main.paint_uniform_color([1, 0, 0])  # Red

        # # Create point cloud for points_to_remove
        # pcd_remove = o3d.geometry.PointCloud()
        # pcd_remove.points = o3d.utility.Vector3dVector(points_to_remove)
        # pcd_remove.paint_uniform_color([0, 1, 0])  # Green
        # # Visualize both point clouds together
        # o3d.visualization.draw_geometries([pcd_main, pcd_remove])

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points_to_remove)
        voxel_grid_to_remove = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size=self.voxel_size*3)
        points_to_remove = voxel_grid_to_remove.check_if_included(o3d.utility.Vector3dVector(self.points))
        points_to_remove = np.array(points_to_remove)
        points_to_remove = points_to_remove > 0
        # points_to_remove = np.logical_not(points_to_remove)  # Invert the boolean array
        # Visualize self.points and points_to_remove in different colors

       

        
        return points_to_remove



    def get_voxel_count(self):
        """
        Get the number of voxels in the voxel grid.
        :return: int
        """
        return len(self.voxel_grid.get_voxels())
    
    def visualize_point_cloud(self, point_size=5.0, point_color=None, voxel_color=None):
        if point_color is None:
            point_color = [1, 0, 0]  # Default red for points
        if voxel_color is None:
            voxel_color = [0, 1, 0]  # Default green for voxel wireframe
            
        # Create visualization window
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        # Add point cloud
        if len(self.points) > 0:
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(self.points)
            # point_cloud.colors = o3d.utility.Vector3dVector(self.colors)
            # point_cloud.paint_uniform_color(point_color)
            vis.add_geometry(point_cloud)
            
            # Set point size
            render_option = vis.get_render_option()
            render_option.point_size = point_size
        
        # Add voxel grid with wireframe rendering
        if self.get_voxel_count() > 0:
            # Get the actual voxels from Open3D
            voxels = self.voxel_grid.get_voxels()
            
            # Create a list to hold all corner points
            all_corners = []
            all_lines = []
            
            # Get the voxel grid origin directly from Open3D voxel grid
            min_bound = self.voxel_grid.get_min_bound()
            
            # Create wireframe representing the voxels
            for voxel in voxels:
                # Get the actual position in world coordinates
                grid_idx = np.array(voxel.grid_index)
                # Calculate the actual position of this voxel in the same coordinate system as the point cloud
                voxel_position = min_bound + grid_idx * self.voxel_size
                
                x, y, z = voxel_position
                corners = [
                    [x, y, z],
                    [x + self.voxel_size, y, z],
                    [x + self.voxel_size, y + self.voxel_size, z],
                    [x, y + self.voxel_size, z],
                    [x, y, z + self.voxel_size],
                    [x + self.voxel_size, y, z + self.voxel_size],
                    [x + self.voxel_size, y + self.voxel_size, z + self.voxel_size],
                    [x, y + self.voxel_size, z + self.voxel_size]
                ]
                
                # Define lines connecting corners to form a cube
                base_idx = len(all_corners)
                all_corners.extend(corners)
                
                indices = [[0, 1], [1, 2], [2, 3], [3, 0],
                           [4, 5], [5, 6], [6, 7], [7, 4],
                           [0, 4], [1, 5], [2, 6], [3, 7]]
                
                # Add the lines for this voxel with the correct base index
                all_lines.extend([[base_idx + i, base_idx + j] for i, j in indices])
            
            if all_corners:
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(all_corners)
                line_set.lines = o3d.utility.Vector2iVector(all_lines)
                line_set.paint_uniform_color(voxel_color)
                # vis.add_geometry(line_set)
            
            # Set voxel rendering to wireframe
            render_option = vis.get_render_option()
            render_option.mesh_show_wireframe = True
        
        # Run the visualizer
        vis.run()
        vis.destroy_window()
        

class GaussianModelTrainer():
    def __init__(self, dyanmic_dataset_manager):
        self.dyanmic_dataset_manager = dyanmic_dataset_manager
        self.point_cloud_data = np.load(self.dyanmic_dataset_manager.pointcloud_path)["data"]
        self.max_cams = self.dyanmic_dataset_manager.train_num_cams
        
        
        
        # Use k=4 to get 3 nearest neighbors (first one is the point itself)
        nbrs = NearestNeighbors(n_neighbors=4).fit(self.point_cloud_data[:, :3])
        distances, _ = nbrs.kneighbors(self.point_cloud_data[:, :3])
        
        # Skip the first column (distance to self is 0)
        distances = distances[:, 1:]
        
        self.avg_distance = np.mean(distances)
        self.min_distance = np.min(distances)
        self.max_distance = np.max(distances)

        
        
        self.gm = GaussianModel(point_cloud_data=self.point_cloud_data, dyanmic_dataset_manager=self.dyanmic_dataset_manager, max_cams=self.max_cams)
        self.ssim = SSIM(data_range=1.0, size_average=False, channel=3).to(self.dyanmic_dataset_manager.device)

        self.flowmodel = ptlflow.get_model("raft", ckpt_path="things")
        self.flowmodel = self.flowmodel.cuda()
        self.flowmodel.eval()
        

    def get_params_and_variables(self):
        return self.gm.params, self.gm.variables
    
    def initialize_optimizer(self, scaling_factor=1.0):
        lrs = {
            'means3D': 0.00016 * self.gm.variables['scene_radius'],
            'rgb_colors': 0.0025,
            'unnorm_rotations': 0.001,
            'logit_opacities': 0.05,
            'log_scales': 0.001,
            'cam_m': 1e-4,
            'cam_c': 1e-4,
        }
        param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]*scaling_factor} for k, v in self.gm.params.items()]
        return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
    def initialize_optimizer_for_selected_points(self, scaling_factor=1.0):
        lrs = {
            'means3D': 0.00016 * self.selected_gm.variables['scene_radius'],
            'rgb_colors': 0.0025,
            'unnorm_rotations': 0.001,
            'logit_opacities': 0.05,
            'log_scales': 0.001,
            'cam_m': 1e-4,
            'cam_c': 1e-4,
        }
        param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]*scaling_factor} for k, v in self.selected_gm.params.items()]
        return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
    
    def l1_loss_v1(self, x, y):
        return torch.abs((x - y)).mean()
    
    def ssim_loss(self, x, y):
        loss = self.ssim(x.unsqueeze(0), y.unsqueeze(0))
        return 1 - loss.mean()
    
    def get_loss_ij(self, timestep, camid):
        curr_data = self.dyanmic_dataset_manager.get_dataset_ij(timestep, camid, train=True)
        rendervar = self.gm.params2rendervar()
        rendervar['means2D'].retain_grad()
        im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
        curr_id = curr_data['id']
        im = torch.exp(self.gm.params['cam_m'][curr_id])[:, None, None] * im + self.gm.params['cam_c'][curr_id][:, None, None]
        loss = 0.8 * self.l1_loss_v1(im, curr_data['im']) + 0.2 * self.ssim_loss(im, curr_data['im'])
        self.gm.variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

        
        seen = radius > 0
        self.gm.variables['max_2D_radius'][seen] = torch.max(radius[seen], self.gm.variables['max_2D_radius'][seen])
        self.gm.variables['seen'] = seen
        return loss, self.gm.variables
    
    def update_variables(self, variables):
        self.gm.variables = variables

    def update_variables_for_selected_points(self, variables):
        self.selected_gm.variables = variables

    def save_model(self, path):
        self.gm.save(path)
    def load_model(self, path):
        self.gm.load(self.dyanmic_dataset_manager, path)

    def get_num_params(self):
        return sum(p.numel() for p in self.gm.params.values())
    def get_model_size(self):
        return sum(p.numel() for p in self.gm.params.values()) * 4 / (1024 ** 2)
    
    def apply_flow_torch(self, image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
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
    def threshold(self, flow, threshold=1.5):
        """
        Convert flow to frame difference.
        
        Args:
            flow: [2, H, W] tensor
            
        Returns:
            frame_diff: [1, H, W] tensor
        """
        # Compute the sum of squares of the flow components
        if flow.shape[0] == 2:
            flow_magnitude_squared = flow[0,:,:]**2 + flow[1,:,:]**2
        else:
            flow_magnitude_squared = flow[0,:,:]

        # Threshold the flow magnitude
        frame_diff = (flow_magnitude_squared > threshold).float()  # Convert to binary mask
        frame_diff = frame_diff.unsqueeze(0)
        # print(frame_diff.shape)
        return frame_diff  # [1, H, W]
    
    def pixels_to_3d(self, cam, im, depth, fg_mask):
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
        H = cam.image_height
        W = cam.image_width
        tanfovx = cam.tanfovx
        tanfovy = cam.tanfovy
        viewmatrix = cam.viewmatrix.squeeze(0)  # [4,4]

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

        campos = cam.campos  # [3]
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

    def remove_low_density_points_in_grid(self, points, grid_size=0.08, k=27):
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

    def get_changes(self, n, cameras_to_select=0.4):
        """
        Get the changes between two timesteps.
        Args:
            n (int): The timestep.
            cameras_to_select (float): The fraction of cameras to select for the change detection.
        Returns:
            all_points (torch.Tensor): The points that are not occluded in both timesteps.
            remove_points (torch.Tensor): The points that are occluded in both timesteps.
            data_to_optimize (torch.Tensor): The points to optimize.
        """
        # Get a list of cameras to select
        # cameras = np.random.rand(int(self.dyanmic_dataset_manager.train_num_cams*cameras_to_select)) * self.dyanmic_dataset_manager.train_num_cams
        # cameras = np.unique(cameras).astype(int)
        # cameras = cameras.tolist()
        cameras = np.array([c for c in range(self.dyanmic_dataset_manager.train_num_cams)])
        num_cameras_to_select = int(self.dyanmic_dataset_manager.train_num_cams * cameras_to_select)
        # Randomly sample num_cameras_to_select cameras from the list of all cameras
        cameras = np.random.choice(cameras, size=num_cameras_to_select, replace=False)
        # print(f"Selected cameras: {cameras}")

        # Get the dataset for the two timesteps
        dataset_prev = self.dyanmic_dataset_manager.get_dataset(n-1, train=True)
        dataset_curr = self.dyanmic_dataset_manager.get_dataset(n, train=True)
        to_return = []
        # Initialize a list to collect all points from different frames
        all_points_list = []
        remove_from_grid_list = []
        for i in range(self.dyanmic_dataset_manager.train_num_cams):
            with torch.no_grad():
                im_prev, radi, depth_prev = self.gm.render(dataset_prev[i]['cam'])
                predictions = self.flowmodel({"images": torch.stack([dataset_curr[i]['im'], dataset_prev[i]['im']], dim=0).unsqueeze(0)})
                flow = predictions["flows"][0, 0]
                predictions = self.flowmodel({"images": torch.stack([ dataset_prev[i]['im'], dataset_curr[i]['im']], dim=0).unsqueeze(0)})
                reverse_flow = predictions["flows"][0, 0]
            prop_depth = self.apply_flow_torch(depth_prev, flow)
            
            frame_difference = torch.abs(dataset_curr[i]['im'] - dataset_prev[i]['im'])
            frame_difference = frame_difference.mean(dim=0, keepdim=True)  # Average over color channels
            frame_difference = self.threshold(frame_difference, threshold=0.5)  # Threshold to create a binary mask

            fg_mask = self.threshold(flow)
            reversed_fg_mask = self.threshold(reverse_flow)
            curr_masked_image =  dataset_curr[i]['im'] * torch.logical_or(fg_mask, frame_difference)

            if i in cameras:
                points_3d_rgb = self.pixels_to_3d(dataset_curr[i]['cam'], dataset_curr[i]['im'], prop_depth, fg_mask)
                all_points_list.append(points_3d_rgb)
            points_to_remove = self.pixels_to_3d(dataset_prev[i]['cam'], dataset_prev[i]['im'], depth_prev, reversed_fg_mask)
            remove_from_grid_list.append(points_to_remove)

            
            
            to_return.append({
                "prev_rendered": im_prev,
                "prev_depth": depth_prev,
                "flow": flow,
                "reverse_flow": reverse_flow,  # Add the reverse flow to the returned dictionary
                "prop_depth": prop_depth,
                "fg_mask": fg_mask,
                "reversed_fg_mask": reversed_fg_mask,
                "frame_difference": frame_difference,  # Add the frame difference to the dictionary
                "curr_masked_image": curr_masked_image  # Add the masked image to the dictionary
            })
        
        all_points = torch.cat(all_points_list, dim=0)
        all_points = self.remove_low_density_points_in_grid(all_points, grid_size=self.avg_distance*2, k=int(len(dataset_curr)*num_cameras_to_select*0.1))

        remove_points = torch.cat(remove_from_grid_list, dim=0)
        remove_points = self.remove_low_density_points_in_grid(remove_points, grid_size=self.avg_distance*2, k=int(len(dataset_curr)*num_cameras_to_select*0.1))

        return all_points, remove_points[:,0:3], to_return
    def initialize_gaussian_for_selected_points(self, all_points):
        """
        Initialize the Gaussian model for the selected points.
        Args:
            all_points (torch.Tensor): The points to optimize.
            data_to_optimize (list): The data to optimize.
            camid (int): The camera ID.
        """
        
        # Initialize the Gaussian model with the selected points
        self.selected_gm = GaussianModel(point_cloud_data=all_points, dyanmic_dataset_manager=self.dyanmic_dataset_manager, cam_m=torch.clone(self.gm.params['cam_m']), cam_c=torch.clone(self.gm.params['cam_c']))

    def train_for_selected_points(self, all_points, data_to_optimize, timestep, camid):

        """
        Train the model for the selected points.
        Args:
            all_points (torch.Tensor): The points to optimize.
            data_to_optimize (list): The data to optimize.
            camid (int): The camera ID.
        Returns:
            loss (torch.Tensor): The loss.
            variables (dict): The variables.
        """
        # Get the dataset for the two timesteps
        curr_data = self.dyanmic_dataset_manager.get_dataset_ij(timestep, camid, train=True)
        rendervar = self.selected_gm.params2rendervar()
        rendervar['means2D'].retain_grad()
        im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
        curr_id = curr_data['id']
        im = torch.exp(self.selected_gm.params['cam_m'][curr_id])[:, None, None] * im + self.selected_gm.params['cam_c'][curr_id][:, None, None]
        loss = self.l1_loss_v1(im, data_to_optimize[camid]['curr_masked_image']) # + 0.2 * self.ssim_loss(im, data_to_optimize[camid]['curr_masked_image'])
        self.selected_gm.variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

        
        seen = radius > 0
        self.selected_gm.variables['max_2D_radius'][seen] = torch.max(radius[seen], self.selected_gm.variables['max_2D_radius'][seen])
        self.selected_gm.variables['seen'] = seen
        return loss, self.selected_gm.variables
    
    def merge_gaussian_for_selected_points(self):
        """
        Merge the Gaussian model for the selected points.
        """
        self.gm.params['means3D'] = torch.cat([self.gm.params['means3D'], self.selected_gm.params['means3D']], dim=0)
        self.gm.params['rgb_colors'] = torch.cat([self.gm.params['rgb_colors'], self.selected_gm.params['rgb_colors']], dim=0)
        self.gm.params['unnorm_rotations'] = torch.cat([self.gm.params['unnorm_rotations'], self.selected_gm.params['unnorm_rotations']], dim=0)
        self.gm.params['logit_opacities'] = torch.cat([self.gm.params['logit_opacities'], self.selected_gm.params['logit_opacities']], dim=0)
        self.gm.params['log_scales'] = torch.cat([self.gm.params['log_scales'], self.selected_gm.params['log_scales']], dim=0)
        self.gm.params['cam_m'] = (self.gm.params['cam_m'] + self.selected_gm.params['cam_m']) / 2
        self.gm.params['cam_c'] = (self.gm.params['cam_c'] + self.selected_gm.params['cam_c']) / 2

        self.gm.variables = {'max_2D_radius': torch.zeros(self.gm.params['means3D'].shape[0]).cuda().float(),
                 'scene_radius': self.gm.scene_radius,
                 'means2D_gradient_accum': torch.zeros(self.gm.params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(self.gm.params['means3D'].shape[0]).cuda().float()}
        
        self.gm.params = {k: torch.nn.Parameter(torch.tensor(v, dtype=torch.float32, device='cuda', requires_grad=True).contiguous()) for k, v in self.gm.params.items()}
    
    def remove_points_from_original(self, remove_points):
        """
        Remove points from the original point cloud using vectorized PyTorch operations.
        Args:
            remove_points (torch.Tensor): The points (or colors) to remove.
        """
        # Check if remove_points is actually just the colors component
        print(f"remove_points shape: {remove_points.shape}")
        if len(remove_points.shape) == 1 or remove_points.shape[1] != 3:
            print(f"Warning: remove_points shape {remove_points.shape} doesn't appear to be 3D coordinates. Skipping removal.")
            return
            
        # Ensure remove_points is a PyTorch tensor on the correct device
        if not isinstance(remove_points, torch.Tensor):
            remove_points = torch.tensor(remove_points, device=self.gm.params['means3D'].device)
        elif remove_points.device != self.gm.params['means3D'].device:
            remove_points = remove_points.to(self.gm.params['means3D'].device)
        
        # Current points
        current_points = self.gm.params['means3D']
        
        # Find the min and max coordinates of the remove_points
        grid_size = self.avg_distance
        min_coords = torch.min(remove_points, dim=0)[0]
        max_coords = torch.max(remove_points, dim=0)[0]
        
        # Extend the bounds slightly to ensure we cover all points
        min_coords = min_coords - grid_size
        max_coords = max_coords + grid_size
        
        # Calculate the number of grid cells in each dimension
        grid_dims = torch.ceil((max_coords - min_coords) / grid_size).long()
        
        # Create indices for the voxel grid
        remove_indices = torch.floor((remove_points - min_coords.unsqueeze(0)) / grid_size).long()
        
        # Ensure indices are within bounds for each dimension separately
        remove_indices[:, 0] = torch.clamp(remove_indices[:, 0], 0, grid_dims[0] - 1)
        remove_indices[:, 1] = torch.clamp(remove_indices[:, 1], 0, grid_dims[1] - 1)
        remove_indices[:, 2] = torch.clamp(remove_indices[:, 2], 0, grid_dims[2] - 1)
        
        # Convert 3D indices to flat indices for the voxel grid
        flat_indices = remove_indices[:, 0] * (grid_dims[1] * grid_dims[2]) + \
                       remove_indices[:, 1] * grid_dims[2] + \
                       remove_indices[:, 2]
        
        # Create a flattened voxel grid and mark occupied cells
        voxel_grid = torch.zeros(int(grid_dims[0] * grid_dims[1] * grid_dims[2]), 
                              dtype=torch.bool, 
                              device=remove_points.device)
        voxel_grid.scatter_(0, flat_indices, torch.ones_like(flat_indices, dtype=torch.bool))
        
        # Calculate grid indices for current points
        point_indices = torch.floor((current_points - min_coords.unsqueeze(0)) / grid_size).long()
        
        # Create a mask for points that are within the grid bounds
        in_bounds = (current_points[:, 0] >= min_coords[0]) & (current_points[:, 0] <= max_coords[0]) & \
                   (current_points[:, 1] >= min_coords[1]) & (current_points[:, 1] <= max_coords[1]) & \
                   (current_points[:, 2] >= min_coords[2]) & (current_points[:, 2] <= max_coords[2])
        
        # Initialize keep_mask with all True
        keep_mask = torch.ones(current_points.shape[0], dtype=torch.bool, device=current_points.device)
        
        # Only consider points that are within the grid bounds
        if torch.any(in_bounds):
            # Get indices of points that are in bounds
            in_bounds_indices = torch.where(in_bounds)[0]
            
            # Get grid indices of in-bounds points
            valid_point_indices = point_indices[in_bounds]
            
            # Clamp indices to ensure they're within grid bounds for each dimension separately
            valid_point_indices[:, 0] = torch.clamp(valid_point_indices[:, 0], 0, grid_dims[0] - 1)
            valid_point_indices[:, 1] = torch.clamp(valid_point_indices[:, 1], 0, grid_dims[1] - 1)
            valid_point_indices[:, 2] = torch.clamp(valid_point_indices[:, 2], 0, grid_dims[2] - 1)
            
            # Convert 3D indices to flat indices
            flat_point_indices = valid_point_indices[:, 0] * (grid_dims[1] * grid_dims[2]) + \
                                valid_point_indices[:, 1] * grid_dims[2] + \
                                valid_point_indices[:, 2]
            
            # Check which points fall in occupied voxels
            points_to_remove = voxel_grid[flat_point_indices]
            
            # Update keep_mask for in-bounds points
            keep_mask[in_bounds_indices[points_to_remove]] = False
        
        # Apply the mask to all parameters
        self.gm.params['means3D'] = torch.nn.Parameter(self.gm.params['means3D'][keep_mask])
        self.gm.params['rgb_colors'] = torch.nn.Parameter(self.gm.params['rgb_colors'][keep_mask])
        self.gm.params['unnorm_rotations'] = torch.nn.Parameter(self.gm.params['unnorm_rotations'][keep_mask])
        self.gm.params['logit_opacities'] = torch.nn.Parameter(self.gm.params['logit_opacities'][keep_mask])
        self.gm.params['log_scales'] = torch.nn.Parameter(self.gm.params['log_scales'][keep_mask])
        
        # Update the variables that depend on the number of points
        self.gm.variables['max_2D_radius'] = torch.zeros(self.gm.params['means3D'].shape[0], device=self.gm.params['means3D'].device).float()
        self.gm.variables['means2D_gradient_accum'] = torch.zeros(self.gm.params['means3D'].shape[0], device=self.gm.params['means3D'].device).float()
        self.gm.variables['denom'] = torch.zeros(self.gm.params['means3D'].shape[0], device=self.gm.params['means3D'].device).float()
        
        print(f"Removed {torch.sum(~keep_mask).item()} points from the original point cloud.")
    def inverse_sigmoid(self, x):
        return torch.log(x / (1 - x))
    def accumulate_mean2d_gradient(self, variables):
        variables['means2D_gradient_accum'][variables['seen']] += torch.norm(
            variables['means2D'].grad[variables['seen'], :2], dim=-1)
        variables['denom'][variables['seen']] += 1
        return variables
    def cat_params_to_optimizer(self, new_params, params, optimizer):
        for k, v in new_params.items():
            group = [g for g in optimizer.param_groups if g['name'] == k][0]
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(v)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(v)), dim=0)
                del optimizer.state[group['params'][0]]
                group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], v), dim=0).requires_grad_(True))
                optimizer.state[group['params'][0]] = stored_state
                params[k] = group["params"][0]
            else:
                group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], v), dim=0).requires_grad_(True))
                params[k] = group["params"][0]
        return params
    def build_rotation(self, q):
        norm = torch.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
        q = q / norm[:, None]
        rot = torch.zeros((q.size(0), 3, 3), device='cuda')
        r = q[:, 0]
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]
        rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
        rot[:, 0, 1] = 2 * (x * y - r * z)
        rot[:, 0, 2] = 2 * (x * z + r * y)
        rot[:, 1, 0] = 2 * (x * y + r * z)
        rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
        rot[:, 1, 2] = 2 * (y * z - r * x)
        rot[:, 2, 0] = 2 * (x * z - r * y)
        rot[:, 2, 1] = 2 * (y * z + r * x)
        rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
        return rot
    def remove_points(self, to_remove, params, variables, optimizer):
        to_keep = ~to_remove
        keys = [k for k in params.keys() if k not in ['cam_m', 'cam_c']]
        for k in keys:
            group = [g for g in optimizer.param_groups if g['name'] == k][0]
            stored_state = optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][to_keep]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][to_keep]
                del optimizer.state[group['params'][0]]
                group["params"][0] = torch.nn.Parameter((group["params"][0][to_keep].requires_grad_(True)))
                optimizer.state[group['params'][0]] = stored_state
                params[k] = group["params"][0]
            else:
                group["params"][0] = torch.nn.Parameter(group["params"][0][to_keep].requires_grad_(True))
                params[k] = group["params"][0]
        variables['means2D_gradient_accum'] = variables['means2D_gradient_accum'][to_keep]
        variables['denom'] = variables['denom'][to_keep]
        variables['max_2D_radius'] = variables['max_2D_radius'][to_keep]
        return params, variables
    def update_params_and_optimizer(self, new_params, params, optimizer):
        for k, v in new_params.items():
            group = [x for x in optimizer.param_groups if x["name"] == k][0]
            stored_state = optimizer.state.get(group['params'][0], None)

            stored_state["exp_avg"] = torch.zeros_like(v)
            stored_state["exp_avg_sq"] = torch.zeros_like(v)
            del optimizer.state[group['params'][0]]

            group["params"][0] = torch.nn.Parameter(v.requires_grad_(True))
            optimizer.state[group['params'][0]] = stored_state
            params[k] = group["params"][0]
        return params
    def adaptive_densification(self, optimizer, i, max_approx_gaussian=3000000, no_clone=False, no_split=False):
        params = self.gm.params
        variables = self.gm.variables
        i = int((2000/100) * i)
        if i <= 5000:
            variables = self.accumulate_mean2d_gradient(variables)
            num_pts = params['means3D'].shape[0]
            grad_thresh = 0.0002

            if (i >= 500) and (i % 100 == 0):
                grads = variables['means2D_gradient_accum'] / variables['denom']
                grads[grads.isnan()] = 0.0
                max_grad = torch.max(grads)
                if 0.9*max_grad > grad_thresh:
                    grad_thresh = (num_pts/max_approx_gaussian)*max_grad.item()
                if not no_clone:
                    to_clone = torch.logical_and(grads >= grad_thresh, (
                                torch.max(torch.exp(params['log_scales']), dim=1).values <= 0.01 * variables['scene_radius']))
                    new_params = {k: v[to_clone] for k, v in params.items() if k not in ['cam_m', 'cam_c']}
                    params = self.cat_params_to_optimizer(new_params, params, optimizer)
                num_pts = params['means3D'].shape[0]
                padded_grad = torch.zeros(num_pts, device="cuda")
                padded_grad[:grads.shape[0]] = grads
                to_split = torch.logical_and(padded_grad >= grad_thresh,
                                            torch.max(torch.exp(params['log_scales']), dim=1).values > 0.01 * variables[
                                                'scene_radius'])

                if not no_split:
                    n = 2  # number to split into
                    new_params = {k: v[to_split].repeat(n, 1) for k, v in params.items() if k not in ['cam_m', 'cam_c']}
                    stds = torch.exp(params['log_scales'])[to_split].repeat(n, 1)
                    means = torch.zeros((stds.size(0), 3), device="cuda")
                    samples = torch.normal(mean=means, std=stds)
                    rots = self.build_rotation(params['unnorm_rotations'][to_split]).repeat(n, 1, 1)
                    new_params['means3D'] += torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
                    new_params['log_scales'] = torch.log(torch.exp(new_params['log_scales']) / (0.8 * n))
                    params = self.cat_params_to_optimizer(new_params, params, optimizer)
                    num_pts = params['means3D'].shape[0]

                    variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda")
                    variables['denom'] = torch.zeros(num_pts, device="cuda")
                    variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda")

                    to_remove = torch.cat((to_split, torch.zeros(n * to_split.sum(), dtype=torch.bool, device="cuda")))
                    params, variables = self.remove_points(to_remove, params, variables, optimizer)

                remove_threshold = 0.25 if i == 5000 else (0.25*(num_pts/max_approx_gaussian))
                to_remove = (torch.sigmoid(params['logit_opacities']) < remove_threshold).squeeze()
                if i >= 1000:
                    big_points_ws = torch.exp(params['log_scales']).max(dim=1).values > 0.1 * variables['scene_radius']
                    to_remove = torch.logical_or(to_remove, big_points_ws)
                params, variables = self.remove_points(to_remove, params, variables, optimizer)

                torch.cuda.empty_cache()

            # if i > 0 and i % 3000 == 0:
            #     new_params = {'logit_opacities': self.inverse_sigmoid(torch.ones_like(params['logit_opacities']) * 0.01)}
            #     params = self.update_params_and_optimizer(new_params, params, optimizer)
        iter = 1
        while params['means3D'].shape[0] > max_approx_gaussian:
            mapped = 1 - (1/(iter*iter)) # mapped 0 to 1
            params = self.selected_gm.params 
            variables = self.selected_gm.variables
            big_points_ws = torch.exp(params['log_scales']).max(dim=1).values > (1/mapped)*self.avg_distance
            # grad_thresh = mapped * max_grad.item()
            # grads = variables['means2D_gradient_accum'] / variables['denom']
            # grads[grads.isnan()] = 0.0
            # big_grads = grads >= grad_thresh

            to_remove = (torch.sigmoid(params['logit_opacities']) < mapped).squeeze()
            to_remove = torch.logical_or(to_remove, big_points_ws)
            # to_remove = torch.logical_or(to_remove, big_grads)
            params, variables = self.remove_points(to_remove, params, variables, optimizer)
            iter += 0.33


            self.gm.params = params
            self.gm.variables = variables
    
    def adaptive_densification_selected(self, optimizer):
        params = self.selected_gm.params
        variables = self.selected_gm.variables
        big_points_ws = torch.exp(params['log_scales']).max(dim=1).values > 0.1 * variables['scene_radius']
        to_remove = (torch.sigmoid(params['logit_opacities']) < 0.7).squeeze()
        to_remove = torch.logical_or(to_remove, big_points_ws)
        params, variables = self.remove_points(to_remove, params, variables, optimizer)
        self.selected_gm.params = params
        self.selected_gm.variables = variables

    # def do_stuff_pre_next_timestep(self, remove_points, optimizer):
    #     self.voxel_model_manager.initialize_points(self.gm.params['means3D'])
    #     remove_points_mask = torch.tensor(self.voxel_model_manager.indices_of_points_to_remove(remove_points), dtype=torch.bool, device=self.gm.params['means3D'].device)
    #     params, variables = self.remove_points(remove_points_mask, self.gm.params, self.gm.variables, optimizer)
    #     self.gm.params = params
    #     self.gm.variables = variables



if __name__ == "__main__":
    model = torch.load("/home/anurag/codes/MV4D_reconstruction/output/start3/dynamic/basketball/final_model.pth")
    vm = VoxelModelManager(voxel_size=0.08)
    vm.initialize_points(model['means3D'])
    vm.visualize_point_cloud()