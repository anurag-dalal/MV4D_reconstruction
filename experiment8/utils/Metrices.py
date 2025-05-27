import numpy as np
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import PeakSignalNoiseRatio
from pytorch_msssim import SSIM, MS_SSIM

class Metrics():
    def __init__(self, dyanmic_dataset_manager, batch_size=1):
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dyanmic_dataset_manager = dyanmic_dataset_manager
        
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim = SSIM(data_range=1.0, size_average=False, channel=3).to(self.device)
        self.ms_ssim = MS_SSIM(data_range=1.0, size_average=False, channel=3).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=False).to(self.device)

    def get_metrics(self, model, timestep, train=True):
        """
        Calculate the metrics for the given parameters and dataset.

        Args:
            params (dict): Parameters to be evaluated.
            curr_timestep (int): Current timestep for evaluation.
            md (dict): Metadata for the dataset.
            dataset_path (str): Path to the dataset.
            batch_size (int): Batch size for evaluation.

        Returns:
            dict: Dictionary containing calculated metrics (L1, PSNR, SSIM, LPIPS, MSSSIM).
        """ 
        if train:
            dataset = self.dyanmic_dataset_manager.get_dataset(timestep, train=True)
        else:
            dataset = self.dyanmic_dataset_manager.get_dataset(timestep, train=False)

        L1_list = []
        PSNR_list = []
        SSIM_list = []
        LPIPS_list = []
        MSSSIM_list = []

        for i in range(len(dataset)):
            gt = dataset[i]['im']
            pred,_, _ = model.gm.render(dataset[i]['cam'])
            pred = torch.clip(pred, min=0.0, max=1.0)
            gt = gt.unsqueeze(0).to(self.device)
            pred = pred.unsqueeze(0).to(self.device)

            L1_list.append(torch.abs(gt - pred).mean().item())
            PSNR_list.append(self.psnr(pred, gt).mean().item())
            SSIM_list.append(self.ssim(pred, gt).mean().item())
            LPIPS_list.append(self.lpips(pred, gt).mean().item())
            MSSSIM_list.append(self.ms_ssim(pred, gt).mean().item())

        # Calculate average metrics
        avg_L1 = sum(L1_list) / len(L1_list) if L1_list else 0
        avg_PSNR = sum(PSNR_list) / len(PSNR_list) if PSNR_list else 0
        avg_SSIM = sum(SSIM_list) / len(SSIM_list) if SSIM_list else 0
        avg_LPIPS = sum(LPIPS_list) / len(LPIPS_list) if LPIPS_list else 0
        avg_MSSSIM = sum(MSSSIM_list) / len(MSSSIM_list) if MSSSIM_list else 0
        # Create all_metrics dictionary with list values
        all_metrics = {
            'L1': L1_list,
            'PSNR': PSNR_list,
            'SSIM': SSIM_list,
            'LPIPS': LPIPS_list,
            'MSSSIM': MSSSIM_list,
            'avg_L1': avg_L1,
            'avg_PSNR': avg_PSNR,
            'avg_SSIM': avg_SSIM,
            'avg_LPIPS': avg_LPIPS,
            'avg_MSSSIM': avg_MSSSIM
        }

        return all_metrics





            


