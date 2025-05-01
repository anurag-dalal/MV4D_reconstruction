import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import torch
import numpy as np
import json
import os
import copy
import time
from PIL import Image
import gc

# Import the necessary functions from experiment1.py
from experiment1 import (
    get_dataset, initialize_params, params2rendervar, 
    Renderer, apply_flow_torch, get_frame_differnce_from_flow,
    process_input_for_flow
)
from ptlflow.utils import flow_utils
import ptlflow

class DynamicSceneViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Dynamic Scene Viewer")
        self.geometry("1600x900")  # Larger initial window size
        
        # Configure styles for larger UI elements
        self.configure_styles()
        
        # Initialize variables
        self.sequence = "basketball"  # Default sequence
        self.exp_name = "exp1"  # Default experiment
        self.current_timestamp = 1   # Start from timestamp 1
        self.current_camera = 0      # Start from camera 0
        
        # Status indicators for the UI
        self.status_var = tk.StringVar(value="Initializing application...")
        
        # Data caching
        self.dataset_cache = {}  # Cache for datasets by timestamp
        self.processed_view_cache = {}  # Cache for processed views
        
        # Create UI first so status can be displayed
        self.create_ui()
        
        # Load metadata and models (can be slow)
        self.after(100, self.initialize_app)
    
    def configure_styles(self):
        """Configure custom styles for larger UI elements"""
        self.style = ttk.Style()
        
        # Configure font sizes
        default_font = ('Helvetica', 12)
        large_font = ('Helvetica', 14)
        button_font = ('Helvetica', 12, 'bold')
        
        # Configure styles
        self.style.configure('TLabel', font=large_font, padding=5)
        self.style.configure('TButton', font=button_font, padding=10)
        self.style.configure('TSpinbox', font=large_font, padding=5)
        self.style.configure('TFrame', padding=10)
        
        # Configure Matplotlib font sizes
        plt.rc('font', size=12)          # controls default text size
        plt.rc('axes', titlesize=14)     # fontsize of the axes title
        plt.rc('axes', labelsize=12)     # fontsize of the x and y labels
        plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
        plt.rc('legend', fontsize=12)    # fontsize of the legend
        plt.rc('figure', titlesize=16)   # fontsize of the figure title
    
    def initialize_app(self):
        """Initialize application in a separate step to keep UI responsive"""
        start_time = time.time()
        self.status_var.set("Loading metadata and models...")
        self.update_idletasks()
        
        # Load metadata
        self.load_metadata()
        
        # Initialize optical flow model (this can be slow)
        self.status_var.set("Loading optical flow model...")
        self.update_idletasks()
        self.initialize_flow_model()
        
        # Load initial data
        self.status_var.set("Loading initial data...")
        self.update_idletasks()
        self.load_initial_data()
        
        elapsed = time.time() - start_time
        self.status_var.set(f"Ready. (Initialization took {elapsed:.1f} seconds)")
    
    def load_metadata(self):
        """Load scene metadata and parameters"""
        # Load metadata for the sequence
        self.metadata = json.load(open(f"/home/anurag/Datasets/dynamic/data/{self.sequence}/train_meta.json", 'r'))
        
        # Get number of timestamps and cameras
        self.num_timestamps = len(self.metadata['fn'])
        self.num_cameras = len(self.metadata['fn'][0])  # Number of cameras at first timestamp
        
        # Update UI elements with correct ranges
        self.timestamp_spinbox.config(to=self.num_timestamps)
        self.camera_spinbox.config(to=self.num_cameras-1)
        
        # Load saved model if it exists
        self.saved_model_path = f"./output/{self.exp_name}/{self.sequence}/params.npz"
        if os.path.exists(self.saved_model_path):
            print(f"Loading model from {self.saved_model_path}")
            self.saved_params = dict(np.load(self.saved_model_path))
            # Initialize parameters from saved model
            self.params, self.variables = self.initialize_model_params()
        else:
            print(f"No saved model found at {self.saved_model_path}")
            self.params, self.variables = initialize_params(self.sequence, self.metadata)
    
    def initialize_flow_model(self):
        """Initialize optical flow model (expensive operation)"""
        self.flow_model = ptlflow.get_model("raft", ckpt_path="things")
        self.flow_model = self.flow_model.cuda()
        self.flow_model.eval()
    
    def initialize_model_params(self, t_idx=1):
        """Initialize model parameters from saved model"""
        params, variables = initialize_params(self.sequence, self.metadata)
        
        # Use the parameters from the saved model for timestamp 0
        params_from_saved = {
            'means3D': torch.tensor(self.saved_params['means3D'][t_idx-1]).cuda().float(),
            'rgb_colors': torch.tensor(self.saved_params['rgb_colors'][t_idx-1]).cuda().float(),
            'seg_colors': torch.tensor(self.saved_params['seg_colors']).cuda().float(),
            'unnorm_rotations': torch.tensor(self.saved_params['unnorm_rotations'][t_idx-1]).cuda().float(),
            'logit_opacities': torch.tensor(self.saved_params['logit_opacities']).cuda().float(),
            'log_scales': torch.tensor(self.saved_params['log_scales']).cuda().float(),
        }
        
        # Update params with values from saved model
        for key in params_from_saved:
            if key in params:
                params[key] = torch.nn.Parameter(params_from_saved[key].contiguous().requires_grad_(True))
        
        return params, variables
    
    def create_ui(self):
        """Create the user interface"""
        # Create frame for controls
        control_frame = ttk.Frame(self, style='TFrame')
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=20)
        
        # Timestamp control
        timestamp_label = ttk.Label(control_frame, text="Timestamp:", style='TLabel')
        timestamp_label.pack(side=tk.LEFT, padx=(0, 10))
        self.timestamp_var = tk.IntVar(value=self.current_timestamp)
        self.timestamp_spinbox = ttk.Spinbox(
            control_frame, 
            from_=1, 
            to=100,  # Will be updated after loading metadata
            textvariable=self.timestamp_var,
            width=5,
            font=('Helvetica', 12)
        )
        self.timestamp_spinbox.pack(side=tk.LEFT, padx=(0, 30))
        
        # Camera control
        camera_label = ttk.Label(control_frame, text="Camera:", style='TLabel')
        camera_label.pack(side=tk.LEFT, padx=(0, 10))
        self.camera_var = tk.IntVar(value=self.current_camera)
        self.camera_spinbox = ttk.Spinbox(
            control_frame, 
            from_=0, 
            to=10,  # Will be updated after loading metadata
            textvariable=self.camera_var,
            width=5,
            font=('Helvetica', 12)
        )
        self.camera_spinbox.pack(side=tk.LEFT, padx=(0, 30))
        
        # Update button
        update_button = ttk.Button(
            control_frame, 
            text="Update View", 
            command=self.on_update_button,
            style='TButton'
        )
        update_button.pack(side=tk.LEFT, padx=(0, 30))
        
        # Sequence selector
        sequence_label = ttk.Label(control_frame, text="Sequence:", style='TLabel')
        sequence_label.pack(side=tk.LEFT, padx=(0, 10))
        self.sequence_var = tk.StringVar(value=self.sequence)
        sequences = ["basketball", "football", "tennis", "softball", "juggle", "boxes"]
        sequence_dropdown = ttk.Combobox(
            control_frame,
            textvariable=self.sequence_var,
            values=sequences,
            state="readonly",
            width=12,
            font=('Helvetica', 12)
        )
        sequence_dropdown.pack(side=tk.LEFT, padx=(0, 20))
        sequence_dropdown.bind('<<ComboboxSelected>>', self.on_sequence_change)
        
        # Status bar with larger font
        status_bar = ttk.Label(
            control_frame, 
            textvariable=self.status_var,
            style='TLabel'
        )
        status_bar.pack(side=tk.RIGHT, padx=10)
        
        # Create matplotlib figure for visualization with larger size
        self.figure = Figure(figsize=(16, 9), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Initialize the subplot grid
        self.setup_subplots()
    
    def on_sequence_change(self, event):
        """Handle sequence change"""
        new_sequence = self.sequence_var.get()
        if new_sequence != self.sequence:
            self.sequence = new_sequence
            self.status_var.set(f"Loading sequence: {self.sequence}...")
            self.update_idletasks()
            
            # Clear caches
            self.dataset_cache = {}
            self.processed_view_cache = {}
            
            # Reset current view
            self.current_timestamp = 1
            self.timestamp_var.set(1)
            self.current_camera = 0
            self.camera_var.set(0)
            
            # Reinitialize with new sequence
            self.after(50, self.initialize_app)
    
    def setup_subplots(self):
        """Create the initial subplot grid"""
        self.figure.clear()
        self.axs = self.figure.subplots(2, 5)
        
        # Increase spacing between subplots
        self.figure.subplots_adjust(wspace=0.3, hspace=0.3)
    
    def on_update_button(self):
        """Process update button click"""
        # Get values from spinboxes
        new_timestamp = self.timestamp_var.get()
        new_camera = self.camera_var.get()
        
        # Check if values have changed
        if new_timestamp != self.current_timestamp or new_camera != self.current_camera:
            self.current_timestamp = new_timestamp
            self.current_camera = new_camera
            self.status_var.set("Processing update...")
            self.update_idletasks()
            
            # Use after() to keep UI responsive during update
            self.after(50, self.update_visualization)
    
    def load_initial_data(self):
        """Load initial datasets to cache"""
        # Preload first two timestamps
        for t_idx in range(min(2, self.num_timestamps)):
            self.get_dataset_for_timestamp(t_idx)
        
        # Process initial view
        self.update_visualization()
    
    def get_dataset_for_timestamp(self, t_idx):
        """Get dataset for a timestamp from cache or load if not available"""
        if t_idx not in self.dataset_cache:
            self.dataset_cache[t_idx] = get_dataset(t_idx, self.metadata, self.sequence)
        return self.dataset_cache[t_idx]
    
    def update_visualization(self):
        """Update the visualization with current data"""
        try:
            # Validate ranges
            if self.current_timestamp < 1:
                self.current_timestamp = 1
                self.timestamp_var.set(1)
            elif self.current_timestamp > self.num_timestamps:
                self.current_timestamp = self.num_timestamps
                self.timestamp_var.set(self.num_timestamps)
            
            if self.current_camera < 0:
                self.current_camera = 0
                self.camera_var.set(0)
            elif self.current_camera >= self.num_cameras:
                self.current_camera = self.num_cameras - 1
                self.camera_var.set(self.num_cameras - 1)
            
            # Convert 1-based timestamp to 0-based index
            t_idx = self.current_timestamp
            self.params, self.variables = self.initialize_model_params(t_idx)
            
            # Get or create cache key
            cache_key = (t_idx, self.current_camera)
            
            start_time = time.time()
            
            # Process view if not in cache
            if cache_key not in self.processed_view_cache:
                self.status_var.set(f"Processing view for t={self.current_timestamp}, cam={self.current_camera}...")
                self.update_idletasks()
                
                # Get datasets for previous and current timestamp
                prev_dataset = self.get_dataset_for_timestamp(max(0, t_idx-1))
                curr_dataset = self.get_dataset_for_timestamp(t_idx)
                
                # Process the current view
                view_data = self.process_view(prev_dataset, curr_dataset)
                
                # Store processed results in cache (only store numpy arrays to save memory)
                self.processed_view_cache[cache_key] = view_data
                
                # Clear GPU memory
                torch.cuda.empty_cache()
                gc.collect()
            
            # Draw the visualization
            self.draw_visualization(self.processed_view_cache[cache_key])
            
            elapsed = time.time() - start_time
            self.status_var.set(f"View updated in {elapsed:.2f} seconds")
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            print(f"Error updating visualization: {e}")
    
    def process_view(self, prev_dataset, curr_dataset):
        """Process the current view to generate visualization data"""
        # Get rendervar for current params
        rendervar = params2rendervar(self.params)
        rendervar['means2D'].retain_grad()
        
        # Get data for selected camera
        camera_idx = self.current_camera
        curr_data = curr_dataset[camera_idx]
        prev_data = prev_dataset[camera_idx]
        
        # Render the scene
        print(prev_data['cam'])
        im, radius, depth = Renderer(raster_settings=prev_data['cam'])(**rendervar)
        
        # Get optical flow
        with torch.no_grad():  # Use no_grad to save memory
            predictions = self.flow_model({"images": torch.stack([curr_data['im'], prev_data['im']], dim=0).unsqueeze(0)})
            flow = predictions["flows"][0, 0]
            
            # Apply flow
            prop_depth = apply_flow_torch(depth, flow)
            fg_mask = get_frame_differnce_from_flow(flow)
        
        # Convert tensors to CPU and numpy for visualization (to save GPU memory)
        prev_im_np = prev_data['im'].permute(1, 2, 0).detach().cpu().numpy()
        curr_im_np = curr_data['im'].permute(1, 2, 0).detach().cpu().numpy()
        prev_depth_np = depth.squeeze(0).detach().cpu().numpy()
        curr_depth_np = prop_depth.squeeze(0).detach().cpu().numpy()
        fg_mask_np = fg_mask.squeeze(0).detach().cpu().numpy() > 0.5
        
        # Process for visualization
        curr_depth_corrected = np.where(fg_mask_np, curr_depth_np, prev_depth_np)
        curr_masked_image = np.where(np.expand_dims(fg_mask_np, axis=2), curr_im_np, 0)
        
        # Process flow for visualization
        flow_np = flow.permute(1, 2, 0).detach().cpu().numpy()
        flow_viz = flow_utils.flow_to_rgb(flow_np)
        
        # Return the processed results as numpy arrays (to save memory)
        return {
            'prev_im_np': prev_im_np,
            'curr_im_np': curr_im_np,
            'prev_depth_np': prev_depth_np,
            'curr_depth_np': curr_depth_np,
            'fg_mask_np': fg_mask_np,
            'curr_depth_corrected': curr_depth_corrected,
            'curr_masked_image': curr_masked_image,
            'flow_viz': flow_viz
        }
    
    def draw_visualization(self, view_data):
        """Draw the visualization with matplotlib using pre-processed data"""
        # Clear previous plots
        for ax in self.axs.flat:
            ax.clear()
        
        # Draw the images
        self.axs[0][0].imshow(view_data['prev_im_np'])
        self.axs[0][0].set_title("Previous Frame", fontsize=14)
        self.axs[0][0].axis("off")
        
        self.axs[0][1].imshow(view_data['prev_depth_np'], cmap="viridis")
        self.axs[0][1].set_title("Previous Depth", fontsize=14)
        self.axs[0][1].axis("off")
        
        self.axs[0][2].imshow(view_data['flow_viz'])
        self.axs[0][2].set_title("Optical Flow", fontsize=14)
        self.axs[0][2].axis("off")
        
        self.axs[0][3].imshow(view_data['fg_mask_np'], cmap="Greys")
        self.axs[0][3].set_title("Foreground Mask", fontsize=14)
        self.axs[0][3].axis("off")
        
        self.axs[0][4].imshow(view_data['curr_masked_image'])
        self.axs[0][4].set_title("Masked Current Frame", fontsize=14)
        self.axs[0][4].axis("off")
        
        self.axs[1][0].imshow(view_data['curr_im_np'])
        self.axs[1][0].set_title("Current Frame", fontsize=14)
        self.axs[1][0].axis("off")
        
        self.axs[1][1].imshow(view_data['curr_depth_np'], cmap="viridis")
        self.axs[1][1].set_title("Current Depth", fontsize=14)
        self.axs[1][1].axis("off")
        
        self.axs[1][2].imshow(view_data['curr_depth_corrected'], cmap="viridis")
        self.axs[1][2].set_title("Corrected Depth", fontsize=14)
        self.axs[1][2].axis("off")
        
        # Add info about current view with larger font
        timestamp_info = f"Timestamp: {self.current_timestamp}/{self.num_timestamps}"
        camera_info = f"Camera: {self.current_camera}/{self.num_cameras-1}"
        sequence_info = f"Sequence: {self.sequence}"
        info_text = f"{timestamp_info}\n{camera_info}\n{sequence_info}"
        
        self.axs[1][3].text(0.5, 0.5, info_text, 
                          horizontalalignment='center',
                          verticalalignment='center',
                          transform=self.axs[1][3].transAxes,
                          fontsize=14,
                          bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        self.axs[1][3].axis("off")
        
        # Draw cache status in the last cell with larger font
        cache_info = f"Dataset Cache: {len(self.dataset_cache)} timestamps\n"
        cache_info += f"View Cache: {len(self.processed_view_cache)} views"
        
        self.axs[1][4].text(0.5, 0.5, cache_info,
                          horizontalalignment='center',
                          verticalalignment='center',
                          transform=self.axs[1][4].transAxes,
                          fontsize=14,
                          bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        self.axs[1][4].axis("off")
        
        # Refresh the canvas
        self.figure.tight_layout()
        self.canvas.draw()

# Function to limit the GPU memory usage for PyTorch
def limit_gpu_memory(fraction=0.7):
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        fraction_memory = int(total_memory * fraction)
        # Set a limit on GPU memory
        torch.cuda.set_per_process_memory_fraction(fraction)
        print(f"GPU memory limited to {fraction*100:.0f}% of available memory")

if __name__ == "__main__":
    # Limit GPU memory usage to prevent crashes
    limit_gpu_memory(0.8)
    
    app = DynamicSceneViewer()
    app.mainloop()