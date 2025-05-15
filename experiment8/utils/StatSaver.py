import pandas as pd
import os

class StatSaver():
    def __init__(self, output_path:str):
        self.output_path = output_path
        self.columns = ['Timestep', 'Num_Iteration', 'Train_L1', 'Train_PSNR', 'Train_SSIM', 'Train_LPIPS', 'Train_MSSSIM', 
                        'Test_L1', 'Test_PSNR', 'Test_SSIM', 'Test_LPIPS', 'Test_MSSSIM', 
                        'Training_Time', 'Num_Params', 'Model_Size']
        self.df = pd.DataFrame(columns=self.columns)
        
    
    def save_stat(self, array):
        data = {}
        for i, col in enumerate(self.columns):
            data[col] = array[i]
        self.df = pd.concat([self.df, pd.DataFrame([data])], ignore_index=True)
        self.df.to_csv(self.output_path, index=False)