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
        flow_magnitude_squared = flow[0,:,:]**2 + flow[1,:,:]**2

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
        cameras = np.random.rand(int(self.dyanmic_dataset_manager.train_num_cams*cameras_to_select)) * self.dyanmic_dataset_manager.train_num_cams
        cameras = np.unique(cameras).astype(int)
        cameras = cameras.tolist()
        print(f"Selected cameras: {cameras}")

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
            fg_mask = self.threshold(flow)
            reversed_fg_mask = self.threshold(reverse_flow)
            curr_masked_image =  dataset_curr[i]['im']* fg_mask

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
                "curr_masked_image": curr_masked_image  # Add the masked image to the dictionary
            })
        
        all_points = torch.cat(all_points_list, dim=0)
        all_points = self.remove_low_density_points_in_grid(all_points, grid_size=self.avg_distance, k=int(len(dataset_curr)*cameras_to_select*0.5))

        remove_points = torch.cat(remove_from_grid_list, dim=0)
        # remove_points = self.remove_low_density_points_in_grid(remove_points, grid_size=self.avg_distance, k=int(len(dataset_curr)*cameras_to_select))

        return all_points, remove_points, to_return
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
        loss = 0.8 * self.l1_loss_v1(im, data_to_optimize[camid]['curr_masked_image'])
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
        Remove points from the original point cloud.
        Args:
            remove_points (torch.Tensor): The points to remove.
        """
        


        

