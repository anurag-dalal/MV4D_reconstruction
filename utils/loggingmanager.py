import pandas as pd
import os

class LoggingManager():
    def __init__(self, output_path:str):
        self.output_path = os.path.join(output_path)
        self.columns = ['Timestep', 'Num_Iteration', 'Train_L1', 'Train_PSNR', 'Train_SSIM', 'Train_LPIPS', 'Train_MSSSIM', 
                        'Test_L1', 'Test_PSNR', 'Test_SSIM', 'Test_LPIPS', 'Test_MSSSIM', 
                        'Training_Time', 'Num_Params', 'Model_Size']
        self.df = pd.DataFrame(columns=self.columns)
        
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
    
    def log_stat(self, array):
        data = {}
        for i, col in enumerate(self.columns):
            data[col] = array[i]
        self.df = self.df.append(data, ignore_index=True)
        self.df.to_csv(os.path.join(self.output_path, "stats.csv"), index=False)