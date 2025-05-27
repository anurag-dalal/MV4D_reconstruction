from utils.DataManager import DynamicGaussianDatasetManager
from utils.DynamicGaussianModel import GaussianModelTrainer, VoxelModelManager
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
    num_iterations = 1000
    start = time.time()
   
    optimizer = model.initialize_optimizer(scaling_factor=15.0)
    if os.path.exists(os.path.join(output_path, f"final_model.pth")):
        model.load_model(os.path.join(output_path, f"final_model.pth"))
    else:
        pbar = tqdm(range(num_iterations), desc=f"Timestep {curr_timestep}")
        running_loss = 0.0
        for i in pbar:
            batch_loss = 0.0
            for j in range(dataset_manager.train_num_cams):
                # Ensure both tensors are on the same device
                loss, variables = model.get_loss_ij(curr_timestep, j)
                model.update_variables(variables)
                batch_loss += loss.item()
                loss.backward()
            with torch.no_grad():
                if i % 10 == 0:
                    model.adaptive_densification(optimizer, j)
                optimizer.step()
                optimizer.zero_grad()
            running_loss = 0.9 * running_loss + 0.1 * batch_loss if i > 0 else batch_loss
            pbar.set_postfix({"loss": f"{running_loss:.6f}"})
        end = time.time()
        model.save_model(os.path.join(output_path, f"final_model.pth"))
        train_all_metrics = metrics_manager.get_metrics(model, curr_timestep, train=True)
        test_all_metrics = metrics_manager.get_metrics(model, curr_timestep, train=False)
        log_stat = [curr_timestep, num_iterations, train_all_metrics['avg_L1'], train_all_metrics['avg_PSNR'], train_all_metrics['avg_SSIM'], train_all_metrics['avg_LPIPS'], train_all_metrics['avg_MSSSIM'],        test_all_metrics['avg_L1'], test_all_metrics['avg_PSNR'], test_all_metrics['avg_SSIM'], test_all_metrics['avg_LPIPS'], test_all_metrics['avg_MSSSIM'],
                    end - start, model.get_num_params(), model.get_model_size()]
        stat_saver.save_stat(log_stat)

    for i in range(1, num_timesteps):
        running_loss = 0.0
        curr_timestep = i
        num_iterations1 = 100
        start = time.time()
        
        
        all_points, remove_points, data_to_optimize = model.get_changes(curr_timestep)
        VM = VoxelModelManager(model.avg_distance*2)
        VM.initialize_points(model.gm.params['means3D'])
        # VM.visualize_point_cloud(point_size=1.0)
        
        # model.initialize_gaussian_for_selected_points(all_points)
        remove_points_indices = VM.indices_of_points_to_remove(remove_points)
        remove_points_mask = torch.tensor(remove_points_indices, dtype=torch.bool, device=model.gm.params['means3D'].device)
        keep_points_mask = ~remove_points_mask
        # VM.visualize_point_cloud(point_size=1.0)
        for k, v in model.gm.params.items():
            if v.shape[0] == keep_points_mask.shape[0]:
                model.gm.params[k] = v[keep_points_mask]
        # VM.visualize_point_cloud(point_size=1.0)
        model.initialize_gaussian_for_selected_points(all_points)
        optimizer = model.initialize_optimizer_for_selected_points(scaling_factor=5.0)
        
        # visulaize_point_cloud_6d_torch_array(all_points)

        pbar = tqdm(range(num_iterations1), desc=f"Optimizing Selected points Timestep {curr_timestep}")
        for j in pbar:
            batch_loss = 0.0
            
            for k in range(dataset_manager.train_num_cams):
                loss, variables = model.train_for_selected_points(all_points, data_to_optimize, i, k)
                model.update_variables_for_selected_points(variables)
                batch_loss += loss.item()
                loss.backward()
            with torch.no_grad():
                if j < num_iterations1//3:
                    model.adaptive_densification_selected(optimizer)
                optimizer.step()
                optimizer.zero_grad()
            running_loss = 0.9 * running_loss + 0.1 * batch_loss if i > 0 else batch_loss
            pbar.set_postfix({"loss": f"{running_loss:.6f}"})
            # with torch.no_grad():
            #     im,_,_ = model.selected_gm.render(dataset_manager.get_dataset_ij(i, 0)['cam'])
            # import matplotlib.pyplot as plt
            # im = im.permute(1, 2, 0)
            #plt.imshow(im.detach().cpu().numpy())
            #plt.show()
        num_iterations2 = 200
        model.merge_gaussian_for_selected_points()
        VM.initialize_points(model.gm.params['means3D'])
        # VM.visualize_point_cloud(point_size=1.0)
        optimizer = model.initialize_optimizer(scaling_factor=3.0)
        # print('After Adding new moved points', model.gm.params['means3D'].shape)
        pbar = tqdm(range(num_iterations2), desc=f"Optimizing All points Timestep {curr_timestep}")    
        for j in pbar:
            batch_loss = 0.0
            for k in range(dataset_manager.train_num_cams):
                loss, variables = model.get_loss_ij(i, k)
                model.update_variables(variables)
                batch_loss += loss.item()
                loss.backward()
            with torch.no_grad():
                # if j < num_iterations2//4:
                #     if j % 15 == 0 and j > 0:
                #         model.adaptive_densification(optimizer, j, no_clone=False, no_split=False)
                #     else:
                #         model.adaptive_densification(optimizer, j, no_clone=True, no_split=True)
                # elif j % 10 == 0:
                #     model.adaptive_densification(optimizer, j, no_clone=True, no_split=True)
                if j < num_iterations2//3 and j>0 and j % 15 == 0:
                    model.adaptive_densification(optimizer, j, no_clone=False, no_split=False)
                else:
                    model.adaptive_densification(optimizer, j, no_clone=True, no_split=True)
                optimizer.step()
                optimizer.zero_grad()
            running_loss = 0.9 * running_loss + 0.1 * batch_loss if i > 0 else batch_loss
            pbar.set_postfix({"loss": f"{running_loss:.6f}"})
        end = time.time()
        # with torch.no_grad():
        #     im,_,_ = model.gm.render(dataset_manager.get_dataset_ij(i, 0)['cam'])
        # im = im.permute(1, 2, 0)
        #plt.imshow(im.detach().cpu().numpy())
        #plt.show()
        train_all_metrics = metrics_manager.get_metrics(model, curr_timestep, train=True)
        test_all_metrics = metrics_manager.get_metrics(model, curr_timestep, train=False)
        log_stat = [curr_timestep, num_iterations1 + num_iterations2, train_all_metrics['avg_L1'], train_all_metrics['avg_PSNR'], train_all_metrics['avg_SSIM'], train_all_metrics['avg_LPIPS'], train_all_metrics['avg_MSSSIM'],        test_all_metrics['avg_L1'], test_all_metrics['avg_PSNR'], test_all_metrics['avg_SSIM'], test_all_metrics['avg_LPIPS'], test_all_metrics['avg_MSSSIM'],
                    end - start, model.get_num_params(), model.get_model_size()]
        stat_saver.save_stat(log_stat)


    


    

    





if __name__ == "__main__":
    exp_name = "start4"
    dataset_name = "dynamic"
    dataset_root_path = "/mnt/c/MyFiles/Datasets/dynamic/data"
    output_path = "/home/anurag/codes/MV4D_reconstruction/output"
    
    sequences = ["basketball", "boxes", "football", "juggle", "softball", "tennis"]
    # sequences = ["boxes"]
    for sequence in sequences:
        dataset_path = os.path.join(dataset_root_path, sequence)
        output_path = os.path.join(output_path, exp_name, dataset_name, sequence)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print(f"Created output directory: {output_path}")
        train(dataset_path, output_path, exp_name)