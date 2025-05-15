import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nerfstudio.field_components.encodings import SHEncoding
from utils.voxel_utils import VoxelGridManager
import json
from diff_gaussian_rasterization_with_depth import GaussianRasterizer as Renderer
from pytorch_msssim import SSIM
from math import exp
from torch.autograd import Variable
import torch.nn.functional as func
class GaussianModel():
    def __init__(self, dyanmic_dataset_manager):
        self.dyanmic_dataset_manager = dyanmic_dataset_manager
        self.point_cloud_data = np.load(self.dyanmic_dataset_manager.pointcloud_path)["data"]
        self.max_cams = self.dyanmic_dataset_manager.train_num_cams
        self.voxel_grid_manager = VoxelGridManager(init_pointcloud_path=self.dyanmic_dataset_manager.pointcloud_path)
        self.avg_distance, self.min_distance, self.max_distance = self.voxel_grid_manager.get_distances()
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
        self.ssim = SSIM(data_range=1.0, size_average=False, channel=3).to(self.dyanmic_dataset_manager.device)
    
    def get_params_and_variables(self):
        return self.params, self.variables
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
    
    def l1_loss_v1(self, x, y):
        return torch.abs((x - y)).mean()
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()
    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window
    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = func.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = func.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = func.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = func.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = func.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    def calc_ssim(self, img1, img2, window_size=11, size_average=True):
        # Use the built-in SSIM implementation from pytorch_msssim instead of custom implementation
        # This should properly handle device placement
        try:
            return self.ssim(img1.unsqueeze(0), img2.unsqueeze(0)).mean()
        except RuntimeError:
            # Fallback to a simpler L1 loss if SSIM calculation fails
            print("Warning: SSIM calculation failed, falling back to L1 loss")
            return 1.0 - self.l1_loss_v1(img1, img2)
    
    def get_loss(self, curr_data):
        losses = {}

        rendervar = self.params2rendervar()
        rendervar['means2D'].retain_grad()
        im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
        curr_id = curr_data['id']
        
        # Ensure tensors are on the same device before operations
        cam_m = self.params['cam_m'][curr_id].to(im.device)
        cam_c = self.params['cam_c'][curr_id].to(im.device)
        
        im = torch.exp(cam_m)[:, None, None] * im + cam_c[:, None, None]
        
        # Make sure target image is on the same device
        target_im = curr_data['im'].to(im.device)
        
        # For now, use only L1 loss to avoid CUDA memory issues
        losses['im'] = self.l1_loss_v1(im, target_im)
        
        self.variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification
        seen = radius > 0
        self.variables['max_2D_radius'][seen] = torch.max(radius[seen], self.variables['max_2D_radius'][seen])
        self.variables['seen'] = seen
        return losses['im'], self.variables


from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import PeakSignalNoiseRatio
from pytorch_msssim import SSIM, MS_SSIM    
class GaussianModelMetrics():
    def __init__(self, gm):
        self.gm = gm
        self.device = gm.dyanmic_dataset_manager.device
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim = SSIM(data_range=1.0, size_average=False, channel=3).to(self.device)
        self.ms_ssim = MS_SSIM(data_range=1.0, size_average=False, channel=3).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=False).to(self.device)

    def get_metrics(self, timestep, dataset):

        L1_list = []
        PSNR_list = []
        SSIM_list = []
        LPIPS_list = []
        MSSSIM_list = []
        with torch.no_grad():
            train_dataset = dataset.get_dataset(timestep, train=True)
            rendervar = self.gm.params2rendervar()
            rendervar['means2D'].retain_grad()
            for i in range(len(train_dataset)):
                im, radius, depth = Renderer(raster_settings=train_dataset[i]['cam'])(**rendervar)
                curr_id = train_dataset[i]['id']
                L1_list.append(self.gm.l1_loss_v1(im, train_dataset[i]['im']).item())
                PSNR_list.append(self.psnr(im, train_dataset[i]['im']).item())
                SSIM_list.append(self.ssim(im, train_dataset[i]['im']).item())
                LPIPS_list.append(self.lpips(im, train_dataset[i]['im']).item())
                MSSSIM_list.append(self.ms_ssim(im, train_dataset[i]['im']).item())
            # Calculate average metrics
        avg_L1 = sum(L1_list) / len(L1_list) if L1_list else 0
        avg_PSNR = sum(PSNR_list) / len(PSNR_list) if PSNR_list else 0
        avg_SSIM = sum(SSIM_list) / len(SSIM_list) if SSIM_list else 0
        avg_LPIPS = sum(LPIPS_list) / len(LPIPS_list) if LPIPS_list else 0
        avg_MSSSIM = sum(MSSSIM_list) / len(MSSSIM_list) if MSSSIM_list else 0

        all_train_metrics = {
                    'avg_L1': avg_L1,
                    'avg_PSNR': avg_PSNR,
                    'avg_SSIM': avg_SSIM,
                    'avg_LPIPS': avg_LPIPS,
                    'avg_MSSSIM': avg_MSSSIM
                }
        L1_list = []
        PSNR_list = []
        SSIM_list = []
        LPIPS_list = []
        MSSSIM_list = []
        with torch.no_grad():
            train_dataset = dataset.get_dataset(timestep, train=False)
            rendervar = self.gm.params2rendervar()
            rendervar['means2D'].retain_grad()
            for i in range(len(train_dataset)):
                im, radius, depth = Renderer(raster_settings=train_dataset[i]['cam'])(**rendervar)
                curr_id = train_dataset[i]['id']
                L1_list.append(self.gm.l1_loss_v1(im, train_dataset[i]['im']).item())
                PSNR_list.append(self.psnr(im, train_dataset[i]['im']).item())
                SSIM_list.append(self.ssim(im, train_dataset[i]['im']).item())
                LPIPS_list.append(self.lpips(im, train_dataset[i]['im']).item())
                MSSSIM_list.append(self.ms_ssim(im, train_dataset[i]['im']).item())
            # Calculate average metrics
        avg_L1 = sum(L1_list) / len(L1_list) if L1_list else 0
        avg_PSNR = sum(PSNR_list) / len(PSNR_list) if PSNR_list else 0
        avg_SSIM = sum(SSIM_list) / len(SSIM_list) if SSIM_list else 0
        avg_LPIPS = sum(LPIPS_list) / len(LPIPS_list) if LPIPS_list else 0
        avg_MSSSIM = sum(MSSSIM_list) / len(MSSSIM_list) if MSSSIM_list else 0
        all_test_metrics = {
            'avg_L1': avg_L1,
            'avg_PSNR': avg_PSNR,
            'avg_SSIM': avg_SSIM,
            'avg_LPIPS': avg_LPIPS,
            'avg_MSSSIM': avg_MSSSIM
        }
        return all_train_metrics, all_test_metrics
