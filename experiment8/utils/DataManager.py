import json
from utils.Helpers import setup_camera
import copy
from PIL import Image
import torch
import numpy as np
import os


class DynamicGaussianDatasetManager():
    def __init__(self, dataset_path, device=None, dataset='dynamic', mode='train_test'):
        
        self.dataset_path = dataset_path
        self.dataset = dataset
        self.mode = mode


        if dataset == 'dynamic':
            if 'train' in mode:                                                  
                self.train_md = json.load(open(f"{dataset_path}/train_meta.json", 'r'))
                self.train_num_cams = len(self.train_md['fn'][0])
            if 'test' in mode:
                self.test_md = json.load(open(f"{dataset_path}/test_meta.json", 'r'))
                self.test_num_cams = len(self.test_md['fn'][0]) 
            self.pointcloud_path = f"{self.dataset_path}/init_pt_cld.npz"
            self.num_timesteps = len(self.train_md['fn'])

        if device  is not None:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    def get_dataset(self, timestep, train=True):
        """
        Get the dataset for a specific timestep and training/testing mode.
        Args:
            timestep (int): The timestep to get the dataset for.
            train (bool): If True, get the training dataset; otherwise, get the testing dataset.
        Returns:
            dataset (list): A list of dictionaries containing camera parameters and images.
                cam (dict): The camera parameters for the specified timestep.
                im (torch.Tensor): The image tensor for the specified timestep.
                id (int): The camera ID for the specified timestep.
                dataset[i] is the i-th camera at timestep.
        """
        if self.dataset == 'dynamic':
            if train:
                md = self.train_md
            else:
                md = self.test_md
            t = timestep
            dataset = []
            for c in range(len(md['fn'][t])):
                w, h, k, w2c = md['w'], md['h'], md['k'][t][c], md['w2c'][t][c]
                cam = setup_camera(w, h, k, w2c, near=1.0, far=100)
                fn = md['fn'][t][c]
                im = np.array(copy.deepcopy(Image.open(f"{self.dataset_path}/ims/{fn}")))
                im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255
                im = im.to(self.device)
                dataset.append({'cam': cam, 'im': im, 'id': c})
            return dataset
    def get_dataset_ij(self, i, j, train=True):
        if self.dataset == 'dynamic':
            if train:
                md = self.train_md
            else:
                md = self.test_md
            t = i
            c = j
            w, h, k, w2c = md['w'], md['h'], md['k'][t][c], md['w2c'][t][c]
            cam = setup_camera(w, h, k, w2c, near=1.0, far=100)
            fn = md['fn'][t][c]
            im = np.array(copy.deepcopy(Image.open(f"{self.dataset_path}/ims/{fn}")))
            im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255
            im = im.to(self.device)
            return {'cam': cam, 'im': im, 'id': c}
        
    def get_changes():
        pass