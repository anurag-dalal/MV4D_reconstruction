from utils.DataManager import DynamicGaussianDatasetManager
from utils.DynamicGaussianModel import GaussianModelTrainer
from utils.Metrices import Metrics
from utils.StatSaver import StatSaver
from utils.Helpers import visulaize_point_cloud_6d_torch_array
import torch
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

def train(dataset_path, output_path, exp_name):
    dataset_manager = DynamicGaussianDatasetManager(dataset_path, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    metrics_manager = Metrics(dataset_manager)
    stat_saver = StatSaver(os.path.join(output_path, "train_stats.csv"))
    num_timesteps = dataset_manager.num_timesteps
    print(f"Number of timesteps: {num_timesteps}")
    print(f"Dataset for timestep 0: {dataset_manager.train_num_cams} cameras")
    model = GaussianModelTrainer(dataset_manager)
    # initial timestep optimization
    curr_timestep = 0
    num_iterations = 300
    start = time.time()
    pbar = tqdm(range(num_iterations), desc=f"Timestep {curr_timestep}")
    running_loss = 0.0
    optimizer = model.initialize_optimizer(scaling_factor=15.0)
    model.load_model(os.path.join(output_path, f"final_model.pth"))
    # for i in pbar:
    #     batch_loss = 0.0
    #     for j in range(dataset_manager.train_num_cams):
    #         # Ensure both tensors are on the same device
    #         loss, variables = model.get_loss_ij(curr_timestep, j)
    #         model.update_variables(variables)
    #         batch_loss += loss.item()
    #         loss.backward()
    #         # with torch.no_grad():
    #     optimizer.step()
    #     optimizer.zero_grad()
    #         # densify
    #     running_loss = 0.9 * running_loss + 0.1 * batch_loss if i > 0 else batch_loss
    #     pbar.set_postfix({"loss": f"{running_loss:.6f}"})
    # end = time.time()
    # model.save_model(os.path.join(output_path, f"final_model.pth"))
    # train_all_metrics = metrics_manager.get_metrics(model, curr_timestep, train=True)
    # test_all_metrics = metrics_manager.get_metrics(model, curr_timestep, train=False)
    # log_stat = [curr_timestep, num_iterations, train_all_metrics['avg_L1'], train_all_metrics['avg_PSNR'], train_all_metrics['avg_SSIM'], train_all_metrics['avg_LPIPS'], train_all_metrics['avg_MSSSIM'],        test_all_metrics['avg_L1'], test_all_metrics['avg_PSNR'], test_all_metrics['avg_SSIM'], test_all_metrics['avg_LPIPS'], test_all_metrics['avg_MSSSIM'],
    #             end - start, model.get_num_params(), model.get_model_size()]
    # stat_saver.save_stat(log_stat)

    for i in range(1, num_timesteps):
        curr_timestep = i
        num_iterations = 300
        start = time.time()
        pbar = tqdm(range(num_iterations), desc=f"Timestep {curr_timestep}")
        optimizer = model.initialize_optimizer(scaling_factor=20.0)
        all_points, remove_points, data_to_optimize = model.get_changes(curr_timestep)
        visulaize_point_cloud_6d_torch_array(all_points)
        # for j in pbar:
        #     for k in range(dataset_manager.train_num_cams):
        break

    


    

    





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