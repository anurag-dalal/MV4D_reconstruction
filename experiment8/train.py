from utils.DataManager import DynamicGaussianDatasetManager
from utils.DynamicGaussianModel import GaussianModelTrainer
import torch
import os
import matplotlib.pyplot as plt
def train(dataset_path, output_path, exp_name):
    dataset_manager = DynamicGaussianDatasetManager(dataset_path, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    num_timesteps = dataset_manager.num_timesteps
    print(f"Number of timesteps: {num_timesteps}")
    current_train_dataset = dataset_manager.get_dataset(0, train=True)
    print(f"Dataset for timestep 0: {len(current_train_dataset)} cameras")
    model = GaussianModelTrainer(dataset_manager)
    

    





if __name__ == "__main__":
    exp_name = "start"
    dataset_name = "dynamic"
    dataset_root_path = "/mnt/c/MyFiles/Datasets/dynamic/data"
    output_path = "/home/anurag/codes/MV4D_reconstruction/output"
    
    sequences = ["basketball", "boxes", "football", "juggle", "softball", "tennis"]
    sequences = ["basketball"]
    for sequence in sequences:
        dataset_path = os.path.join(dataset_root_path, sequence)
        output_path = os.path.join(output_path, exp_name, dataset_name, sequence)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        train(dataset_path, output_path, exp_name)