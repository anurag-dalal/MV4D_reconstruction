import numpy as np
import torch
from utils.train_utils import get_dataset_eval, params2rendervar
from utils.train_utils import l1_loss_v1, calc_psnr, calc_ssim
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import PeakSignalNoiseRatio
from pytorch_msssim import SSIM, MS_SSIM

def get_metrics_batch(gt_batch, pred_batch):
    """
    Calculate metrics for batches of ground truth and prediction images.
    
    Args:
        gt_batch (torch.Tensor): Ground truth images of shape [B, 3, H, W]
        pred_batch (torch.Tensor): Predicted images of shape [B, 3, H, W]
        
    Returns:
        dict: Dictionary containing calculated metrics (L1, PSNR, SSIM, LPIPS, MSSSIM)
    """
    device = pred_batch.device
    
    # Initialize metric calculators
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = SSIM(data_range=1.0, size_average=False, channel=3).to(device)
    ms_ssim = MS_SSIM(data_range=1.0, size_average=False, channel=3).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=False).to(device)
    
    # Calculate L1 loss
    l1_values = torch.abs(gt_batch - pred_batch).mean(dim=[1, 2, 3])
    
    # Calculate PSNR
    psnr_values = psnr(pred_batch, gt_batch)
    
    # Calculate SSIM
    ssim_values = ssim(pred_batch, gt_batch)
    
    # Calculate LPIPS
    lpips_values = lpips(pred_batch, gt_batch)
    
    # Calculate MS-SSIM
    msssim_values = ms_ssim(pred_batch, gt_batch)
    
    # Return all metrics as a dictionary
    metrics = {
        'L1': l1_values,
        'PSNR': psnr_values,
        'SSIM': ssim_values,
        'LPIPS': lpips_values,
        'MSSSIM': msssim_values
    }
    
    return metrics

def get_metrics(params, curr_timestep, md, dataset_path, batch_size=1):
    """
    Calculate the metrics for the given parameters and dataset.

    Args:
        params (dict): The parameters of the model.
        curr_timestep (int): The current timestep.
        md (object): The model data object.
        dataset_path (str): The path to the dataset.
        batch_size (int, optional): Batch size for processing. Defaults to 1.

    Returns:
        tuple: A tuple containing the calculated metrics (L1, PSNR, SSIM, LPIPS, MSSSIM).
    """
    # Get the dataset
    dataset = get_dataset_eval(curr_timestep, md, dataset_path)
    
    # Initialize tensors to store metrics
    L1_list = []
    PSNR_list = []
    SSIM_list = []
    LPIPS_list = []
    MSSSIM_list = []
    
    # Determine device to use
    device = None
    
    # Process dataset in batches
    dataset_size = len(dataset)
    for batch_start in range(0, dataset_size, batch_size):
        batch_end = min(batch_start + batch_size, dataset_size)
        current_batch_size = batch_end - batch_start
        
        # Create tensors to hold batch data
        gt_batch = []
        pred_batch = []
        
        # Process each item in the batch
        for i in range(batch_start, batch_end):
            # Get ground truth image
            gt = dataset[i]['im']  # Expected shape: [3, H, W]
            
            # Get rendered prediction
            rendervar = params2rendervar(params)
            rendervar['means2D'].retain_grad()
            pred, _, _ = Renderer(raster_settings=dataset[i]['cam'])(**rendervar)
            
            # Set device if not set yet
            if device is None:
                device = pred.device
            
            # Ensure tensors are on the same device
            gt = gt.to(device)
            
            # Make sure tensors have the correct shape
            if pred.shape != gt.shape:
                raise ValueError(f"Prediction shape {pred.shape} doesn't match ground truth shape {gt.shape}")
            
            # Add to batch lists
            gt_batch.append(gt.unsqueeze(0))    # Add batch dimension
            pred_batch.append(pred.unsqueeze(0))
        
        # Stack tensors along batch dimension
        gt_batch = torch.cat(gt_batch, dim=0)
        pred_batch = torch.cat(pred_batch, dim=0)
        
        # Calculate metrics for the current batch
        metrics = get_metrics_batch(gt_batch, pred_batch)
        
        # Add batch metrics to overall lists - ensure all are handled as lists
        L1_list.extend(metrics['L1'].detach().cpu().tolist())
        
        # Handle scalar vs tensor outputs properly
        if metrics['PSNR'].dim() == 0:  # scalar
            PSNR_list.append(metrics['PSNR'].item())
        else:  # tensor
            PSNR_list.extend(metrics['PSNR'].detach().cpu().tolist())
            
        if metrics['SSIM'].dim() == 0:  # scalar
            SSIM_list.append(metrics['SSIM'].item())
        else:  # tensor
            SSIM_list.extend(metrics['SSIM'].detach().cpu().tolist())
            
        if metrics['LPIPS'].dim() == 0:  # scalar
            LPIPS_list.append(metrics['LPIPS'].item())
        else:  # tensor
            LPIPS_list.extend(metrics['LPIPS'].detach().cpu().tolist())
            
        if metrics['MSSSIM'].dim() == 0:  # scalar
            MSSSIM_list.append(metrics['MSSSIM'].item())
        else:  # tensor
            MSSSIM_list.extend(metrics['MSSSIM'].detach().cpu().tolist())
    
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

    return avg_L1, avg_PSNR, avg_SSIM, avg_LPIPS, avg_MSSSIM, all_metrics