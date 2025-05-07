import torch
import numpy as np
import os


class datamanager:
    def __init__(self, data_path:str, is_numpy:bool=True, from_pointcloud=False, device=None):
        self.params_from_saved = {}
        self.is_numpy = is_numpy
        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if from_pointcloud:
            self.read_data_from_pointcloud(data_path)
        else:
            self.read_data(data_path)
        self.data_path = data_path
        
    def read_data_from_pointcloud(self, data_path:str):
        init_pt_cld = np.load(f"{data_path}/init_pt_cld.npz")["data"]
        
    def read_data(self, data_path:str):
        if self.is_numpy:
            saved_params = dict(np.load(data_path))
            for k,v in saved_params.items():
                self.params_from_saved[k] = torch.from_numpy(v).to(self.device)
            
        else:
            self.params_from_saved = torch.load(data_path)
        
        moved_tensor_dict = {}
        for key, value in self.params_from_saved.items():
            if isinstance(value, torch.Tensor):
                moved_tensor_dict[key] = value.to(self.device)
            else:
                moved_tensor_dict[key] = value  # Keep non-tensor values as they are

        # Replace the original dictionary with the one on the target device
        self.params_from_saved = moved_tensor_dict 
        del moved_tensor_dict
    
    def read_data_at_timestep(self, timestep:int):
        if self.is_numpy:
            saved_params = dict(np.load(self.data_path))
            for k,v in saved_params.items():
                self.params_from_saved[k] = torch.from_numpy(v[timestep]).to(self.device)
            
        else:
            self.params_from_saved = torch.load(self.data_path)
        
        moved_tensor_dict = {}
        for key, value in self.params_from_saved.items():
            if isinstance(value, torch.Tensor):
                moved_tensor_dict[key] = value[timestep].to(self.device)
            else:
                moved_tensor_dict[key] = value

    def get_distances(self):
        if not self.params_from_saved:
            raise ValueError("No data loaded. Please load data first.")
        distances, _ = self.k_nearest_sklearn(self.params_from_saved['means3D'].data, 3)
        avg =  distances.mean().item()
        min =  distances.min().item()
        max =  distances.max().item()
        return avg, min, max
        
    def get_data(self):
        return self.params_from_saved

    def clear_data(self):
        self.params_from_saved.clear()
        
    def k_nearest_sklearn(self, x: torch.Tensor, k: int):
        """
            Find k-nearest neighbors using sklearn's NearestNeighbors.
        x: The data tensor of shape [num_samples, num_features]
        k: The number of neighbors to retrieve
        """
        # Convert tensor to numpy array
        x_np = x.cpu().numpy()

        # Build the nearest neighbors model
        from sklearn.neighbors import NearestNeighbors

        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

        # Find the k-nearest neighbors
        distances, indices = nn_model.kneighbors(x_np)

        # Exclude the point itself from the result and return
        return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)
    
    def save_data(self, data_path:str):
        torch.save(self.params_from_saved, data_path)