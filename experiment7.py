import os
from utils.datamanager import DynamicGaussianDatasetManager
import pandas as pd
from utils.loggingmanager import LoggingManager
import torch
import ptlflow
from utils.voxel_utils import VoxelGridManager
from model.gaussian_model import GaussianModel, GaussianModelMetrics
import time
from tqdm import tqdm

def train(dataset_path, output_path, exp_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    visualize = False
    logging_maneger = LoggingManager(output_path)    
    dyanmic_dataset_manager = DynamicGaussianDatasetManager(dataset_path, device)
    
    flowmodel = ptlflow.get_model("raft", ckpt_path="things")
    flowmodel = flowmodel.to(device)
    flowmodel.eval()
    
    gm = GaussianModel(dyanmic_dataset_manager)
    metrices = GaussianModelMetrics(gm)
    num_timesteps = dyanmic_dataset_manager.num_timesteps
    train_num_cams = dyanmic_dataset_manager.train_num_cams
    test_num_cams = dyanmic_dataset_manager.test_num_cams

    # initial timestep optimization
    curr_timestep = 0
    num_iterations = 300
    prev_params, prev_variables = gm.get_params_and_variables()
    zeroth_optimizer = gm.initialize_optimizer(scaling_factor=15.0)
    zeroth_dataset = dyanmic_dataset_manager.get_dataset(curr_timestep, train=True)

    start = time.time()
    pbar = tqdm(range(num_iterations), desc=f"Timestep {curr_timestep}")
    running_loss = 0.0
    for i in pbar:
        batch_loss = 0.0
        for j in range(train_num_cams):
            loss, prev_variables = gm.get_loss(zeroth_dataset[j])
            batch_loss += loss.item()
            loss.backward()
        running_loss = 0.9 * running_loss + 0.1 * batch_loss if i > 0 else batch_loss
        pbar.set_postfix({"loss": f"{running_loss:.6f}"})
        zeroth_optimizer.step()
        zeroth_optimizer.zero_grad()
        # if i % 10 == 0:
        # zeroth_params, zeroth_variable = zeroth_densify(zeroth_params, zeroth_variable, zeroth_optimizer, i, avg_distance * 3, max_iters=num_iter_per_timestep)
    end = time.time()
    train_metices, test_metices = metrices.get_metrics(curr_timestep, dyanmic_dataset_manager)
    print(f"Train metices: {train_metices}")
    print(f"Test metices: {test_metices}")



        





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