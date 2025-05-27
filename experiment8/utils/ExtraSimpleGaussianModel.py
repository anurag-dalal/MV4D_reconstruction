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


class ExtraSimpleGaussianModel():

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
    
    def __init__(self, point_cloud_data, cams):
        if isinstance(point_cloud_data, torch.Tensor):
            point_cloud_data = point_cloud_data.cpu().numpy()
        sq_dist, _ = self.o3d_knn(point_cloud_data[:, :3], 3)
        mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001)
        max_cams = len(cams)

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
        self.cam_centres = []
        for i in range(max_cams):
            self.cam_centres.append(cams[i].campos.cpu().numpy())
        self.cam_centers = np.array(self.cam_centres)
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
    
    def initialize_optimizer(self, scaling_factor=1.0):
        lrs = {
            'means3D': 0.00016 * self.variables['scene_radius'],
            'rgb_colors': 0.0025,
            'unnorm_rotations': 0.001,
            'logit_opacities': 0.05,
            'log_scales': 0.001,
            'cam_m': 1e-4,
            'cam_c': 1e-4,
        }
        param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]*scaling_factor} for k, v in self.params.items()]
        return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
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
    def adaptive_densification(self, optimizer):
        params = self.params
        variables = self.variables
        big_points_ws = torch.exp(params['log_scales']).max(dim=1).values > 0.05 * variables['scene_radius']
        to_remove = (torch.sigmoid(params['logit_opacities']) < 0.5).squeeze()
        to_remove = torch.logical_or(to_remove, big_points_ws)
        params, variables = self.remove_points(to_remove, params, variables, optimizer)
        self.params = params
        self.variables = variables