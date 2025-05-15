import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nerfstudio.field_components.encodings import SHEncoding
from diff_gaussian_rasterization_with_depth import GaussianRasterizer as Renderer
from pytorch_msssim import SSIM
from math import exp
from sklearn.neighbors import NearestNeighbors

class GaussianModel():
    def __init__(self, params, variables):
        self.params = params
        self.variables = variables
    def __init__(self, point_cloud_data, max_cams, avg_distance):
        self.params = {
            'means3D': self.point_cloud_data[:, :3],
            'rgb_colors': self.point_cloud_data[:, 3:6],
            'unnorm_rotations': np.tile([1, 0, 0, 0], (self.point_cloud_data.shape[0], 1)),
            'logit_opacities': np.ones((self.point_cloud_data.shape[0], 1)),
            'log_scales': np.tile(np.log(np.sqrt(self.avg_distance))[..., None], (1, 3)),
            'cam_m': np.zeros((self.max_cams, 3)),
            'cam_c': np.zeros((self.max_cams, 3)),
        }
        self.params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in
              self.params.items()}
        self.cam_centers = np.linalg.inv(self.dyanmic_dataset_manager.train_md['w2c'][0])[:, :3, 3]  # Get scene radius
        self.scene_radius = 1.1 * np.max(np.linalg.norm(self.cam_centers - np.mean(self.cam_centers, 0)[None], axis=-1))
        self.variables = {'max_2D_radius': torch.zeros(self.params['means3D'].shape[0]).cuda().float(),
                 'scene_radius': self.scene_radius,
                 'means2D_gradient_accum': torch.zeros(self.params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(self.params['means3D'].shape[0]).cuda().float()}
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
        rendervar['means2D'].retain_grad()
        im, radius, depth = Renderer(raster_settings=cam)(**rendervar)
        return im, radius, depth

    
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

        
        
        self.gm = GaussianModel(self.point_cloud_data, self.max_cams, self.avg_distance)
        self.ssim = SSIM(data_range=1.0, size_average=False, channel=3).to(self.dyanmic_dataset_manager.device)

    def get_params_and_variables(self):
        return self.gm.params, self.gm.variables
    
    def get_loss(self, timestep):
        
    
