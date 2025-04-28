import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import json
import os
import torch
import sys
from experiment1 import get_dataset, frame_difference_background_removal, warp_depth_with_flow, get_frame_differnce_from_flow
from helpers import params2rendervar
import matplotlib.pyplot as plt
from ptlflow.utils import flow_utils
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
import copy

class ImageViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("MV4D Dataset Viewer")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.current_timestep = 0
        self.current_camera_idx = 0
        self.sequences = self.get_available_sequences()
        self.current_sequence = self.sequences[0] if self.sequences else None
        self.metadata = None
        self.dataset = None
        self.current_image_type = "rgb"  # Default image type
        
        # Create UI elements
        self.create_ui()
        
        # Load initial data
        if self.current_sequence:
            self.load_metadata()
            self.load_dataset()
            self.display_current_image()
    
    def get_available_sequences(self):
        """Get available sequences from the dataset directory"""
        try:
            base_path = "/home/anurag/Datasets/dynamic/data"
            sequences = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
            return sorted(sequences)
        except Exception as e:
            print(f"Error loading sequences: {e}")
            return []
    
    def load_metadata(self):
        """Load metadata for the current sequence"""
        try:
            metadata_path = f"/home/anurag/Datasets/dynamic/data/{self.current_sequence}/train_meta.json"
            self.metadata = json.load(open(metadata_path, 'r'))
            self.num_timesteps = len(self.metadata['fn'])
            self.update_status(f"Loaded metadata for {self.current_sequence}: {self.num_timesteps} timesteps")
        except Exception as e:
            self.update_status(f"Error loading metadata: {e}")
            self.metadata = None
    
    def load_dataset(self):
        """Load dataset for the current timestep"""
        if not self.metadata:
            self.update_status("No metadata loaded")
            return
        
        try:
            # Use CUDA if available, otherwise use CPU
            if torch.cuda.is_available():
                self.dataset = get_dataset(self.current_timestep, self.metadata, self.current_sequence)
                self.process_dataset_images()
            else:
                # We need to modify how the get_dataset function works for CPU-only mode
                # This is a simplified version that only loads the image paths
                self.dataset = []
                for c in range(len(self.metadata['fn'][self.current_timestep])):
                    fn = self.metadata['fn'][self.current_timestep][c]
                    img_path = f"/home/anurag/Datasets/dynamic/data/{self.current_sequence}/ims/{fn}"
                    self.dataset.append({'path': img_path, 'id': c})
            
            self.num_cameras = len(self.dataset)
            self.update_status(f"Loaded dataset for timestep {self.current_timestep}: {self.num_cameras} cameras")
        except Exception as e:
            self.update_status(f"Error loading dataset: {e}")
            self.dataset = None
    
    def process_dataset_images(self):
        """Process the dataset to extract different image types using Renderer for depth and ptlflow for optical flow"""
        if not torch.cuda.is_available():
            self.update_status("CUDA not available. Cannot process advanced image types.")
            return
        
        # Show progress dialog
        progress_window = self.show_progress_dialog(
            title="Processing Images", 
            message=f"Processing images for timestep {self.current_timestep}..."
        )
        
        # Process images if we have the next timestep available
        try:
            # Import necessary modules for flow computation
            import ptlflow
            from torch.nn import functional as F
            
            # Load flow model if needed
            if not hasattr(self, 'flow_model'):
                self.update_status("Loading optical flow model...")
                progress_window.title("Loading Flow Model")
                progress_window.update()
                try:
                    self.flow_model = ptlflow.get_model("raft", pretrained_ckpt="things")
                    self.flow_model = self.flow_model.cuda()
                    self.flow_model.eval()
                    self.update_status("Optical flow model loaded successfully")
                except Exception as e:
                    self.update_status(f"Error loading flow model: {str(e)}. Using placeholder flow.")
                    self.flow_model = None
            
            # For each timestep, we need both current and previous timestep data for flow calculation
            prev_timestep = max(0, self.current_timestep - 1)
            if prev_timestep != self.current_timestep:
                self.update_status(f"Loading previous timestep data (timestep {prev_timestep})...")
                progress_window.title("Loading Previous Timestep")
                progress_window.update()
                prev_dataset = get_dataset(prev_timestep, self.metadata, self.current_sequence)
            else:
                # If we're at timestep 0, use the same dataset for both
                self.update_status("Using current timestep as previous for flow calculation")
                prev_dataset = copy.deepcopy(self.dataset)
            
            # Load saved model parameters if available
            saved_model_path = f"./output/exp1/{self.current_sequence}/params.npz"
            rendervar = None
            if os.path.exists(saved_model_path):
                progress_window.title("Loading Model Parameters")
                progress_window.update()
                rendervar = self.load_model_parameters(saved_model_path)
            else:
                self.update_status("No model parameters found. Using placeholder rendering.")
            
            # Process each camera view
            for i in range(len(self.dataset)):
                # Update progress dialog
                progress_window.title(f"Processing Camera {i+1}/{len(self.dataset)}")
                progress_window.update()
                
                curr_data = self.dataset[i]
                prev_data = prev_dataset[i]
                
                # Get RGB image and segmentation
                im = curr_data['im']
                seg = curr_data['seg']
                
                # Render depth if we have model parameters
                if rendervar is not None:
                    try:
                        # Use the Renderer to get depth
                        self.update_status(f"Rendering depth for camera {i}...")
                        rendered_im, radius, depth = Renderer(raster_settings=curr_data['cam'])(**rendervar)
                        self.update_status(f"Depth rendering complete for camera {i}")
                    except Exception as e:
                        self.update_status(f"Error rendering depth: {str(e)}. Using placeholder depth.")
                        depth = torch.zeros((1, im.shape[1], im.shape[2]), device=im.device)
                else:
                    # Use placeholder depth if no model parameters
                    depth = torch.zeros((1, im.shape[1], im.shape[2]), device=im.device)
                
                # Compute optical flow between previous and current frame
                flow = torch.zeros((2, im.shape[1], im.shape[2]), device=im.device)
                if self.flow_model is not None:
                    try:
                        with torch.no_grad():
                            self.update_status(f"Computing optical flow for camera {i}...")
                            predictions = self.flow_model({
                                "images": torch.stack([prev_data['im'], curr_data['im']], dim=0).unsqueeze(0)
                            })
                            flow = predictions["flows"][0, 0]  # [2, H, W]
                            self.update_status(f"Flow computation complete for camera {i}")
                    except Exception as e:
                        self.update_status(f"Error computing flow: {str(e)}. Using placeholder flow.")
                        flow = torch.zeros((2, im.shape[1], im.shape[2]), device=im.device)
                
                # Extract foreground mask using flow
                try:
                    fg_mask = get_frame_differnce_from_flow(flow)
                except Exception as e:
                    self.update_status(f"Error extracting foreground mask: {str(e)}. Using placeholder mask.")
                    fg_mask = torch.zeros((1, im.shape[1], im.shape[2]), device=im.device)
                
                # Extract foreground using background subtraction
                try:
                    _, foreground = frame_difference_background_removal(prev_data['im'], curr_data['im'])
                except Exception as e:
                    self.update_status(f"Error extracting foreground: {str(e)}. Using placeholder foreground.")
                    foreground = torch.zeros_like(im)
                
                # Propagate depth using flow
                try:
                    prop_depth = warp_depth_with_flow(depth, flow)
                except Exception as e:
                    self.update_status(f"Error propagating depth: {str(e)}. Using placeholder propagated depth.")
                    prop_depth = torch.zeros_like(depth)
                
                # Compute masked depth
                masked_depth = depth * fg_mask
                
                # Store all image types in the dataset
                curr_data['flow'] = flow
                curr_data['depth'] = depth
                curr_data['prop_depth'] = prop_depth
                curr_data['fg_mask'] = fg_mask
                curr_data['foreground'] = foreground
                curr_data['masked_depth'] = masked_depth
            
            self.update_status(f"Image processing completed for timestep {self.current_timestep}")
            
        except Exception as e:
            self.update_status(f"Error processing images: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            # Close progress dialog
            if progress_window:
                progress_window.destroy()
    
    def display_current_image(self):
        """Display the current image based on selected image type"""
        if not self.dataset or self.current_camera_idx >= len(self.dataset):
            self.update_status("No dataset loaded or invalid camera index")
            return
        
        try:
            data = self.dataset[self.current_camera_idx]
            
            # Get image based on selected type
            if torch.cuda.is_available():
                # Get tensor image based on image type
                if self.current_image_type == "rgb":
                    image_tensor = data['im']
                    image_np = (image_tensor.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                    image = Image.fromarray(image_np)
                elif self.current_image_type == "segmentation":
                    image_tensor = data['seg']
                    image_np = (image_tensor.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                    image = Image.fromarray(image_np)
                elif self.current_image_type == "depth":
                    image_tensor = data['depth']
                    image_np = image_tensor.squeeze(0).detach().cpu().numpy()
                    # Normalize depth for visualization
                    if image_np.max() > image_np.min():
                        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
                    image_np = (image_np * 255).astype(np.uint8)
                    # Create a colormap for depth
                    image_np = plt.cm.viridis(image_np)[:, :, :3]
                    image_np = (image_np * 255).astype(np.uint8)
                    image = Image.fromarray(image_np)
                elif self.current_image_type == "flow":
                    image_tensor = data['flow']
                    image_np = image_tensor.permute(1, 2, 0).detach().cpu().numpy()
                    # Convert flow to RGB using flow_utils
                    image_np = flow_utils.flow_to_rgb(image_np)
                    image = Image.fromarray(image_np)
                elif self.current_image_type == "prop_depth":
                    image_tensor = data['prop_depth']
                    image_np = image_tensor.squeeze(0).detach().cpu().numpy()
                    # Normalize depth for visualization
                    if image_np.max() > image_np.min():
                        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
                    image_np = (image_np * 255).astype(np.uint8)
                    # Create a colormap for depth
                    image_np = plt.cm.viridis(image_np)[:, :, :3]
                    image_np = (image_np * 255).astype(np.uint8)
                    image = Image.fromarray(image_np)
                elif self.current_image_type == "fg_mask":
                    image_tensor = data['fg_mask']
                    image_np = image_tensor.squeeze(0).detach().cpu().numpy()
                    image_np = (image_np * 255).astype(np.uint8)
                    image = Image.fromarray(image_np)
                elif self.current_image_type == "foreground":
                    image_tensor = data['foreground']
                    image_np = (image_tensor.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                    image = Image.fromarray(image_np)
                elif self.current_image_type == "masked_depth":
                    image_tensor = data['masked_depth']
                    image_np = image_tensor.squeeze(0).detach().cpu().numpy()
                    # Normalize depth for visualization
                    if image_np.max() > image_np.min():
                        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
                    image_np = (image_np * 255).astype(np.uint8)
                    # Create a colormap for depth
                    image_np = plt.cm.viridis(image_np)[:, :, :3]
                    image_np = (image_np * 255).astype(np.uint8)
                    image = Image.fromarray(image_np)
                else:
                    # Default to RGB
                    image_tensor = data['im']
                    image_np = (image_tensor.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                    image = Image.fromarray(image_np)
            else:
                # If using CPU, only display RGB images from path
                image_path = data['path']
                image = Image.open(image_path)
                self.current_image_type = "rgb"  # Force RGB for CPU mode
            
            # Resize image to fit the canvas while maintaining aspect ratio
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:  # Ensure canvas has been drawn
                image = self.resize_image(image, canvas_width, canvas_height)
            
            # Convert to PhotoImage and display
            self.photo_image = ImageTk.PhotoImage(image)
            self.image_canvas.create_image(
                canvas_width // 2, canvas_height // 2, 
                anchor=tk.CENTER, image=self.photo_image
            )
            
            # Update info text
            camera_id = data['id'] if 'id' in data else self.current_camera_idx
            self.info_text.set(
                f"Sequence: {self.current_sequence} | "
                f"Timestep: {self.current_timestep} | "
                f"Camera: {camera_id} | "
                f"Type: {self.current_image_type}"
            )
        
        except Exception as e:
            self.update_status(f"Error displaying image: {e}")
    
    def set_image_type(self, image_type):
        """Set the current image type and update display"""
        self.current_image_type = image_type
        self.display_current_image()
    
    def resize_image(self, image, target_width, target_height):
        """Resize image to fit target dimensions while preserving aspect ratio"""
        width, height = image.size
        ratio = min(target_width / width, target_height / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def next_camera(self):
        """Show the next camera view"""
        if not self.dataset:
            return
        
        self.current_camera_idx = (self.current_camera_idx + 1) % len(self.dataset)
        self.display_current_image()
    
    def prev_camera(self):
        """Show the previous camera view"""
        if not self.dataset:
            return
        
        self.current_camera_idx = (self.current_camera_idx - 1) % len(self.dataset)
        self.display_current_image()
    
    def next_timestep(self):
        """Load the next timestep"""
        if not self.metadata:
            return
        
        self.current_timestep = (self.current_timestep + 1) % self.num_timesteps
        self.current_camera_idx = 0  # Reset camera index
        self.load_dataset()
        self.display_current_image()
    
    def prev_timestep(self):
        """Load the previous timestep"""
        if not self.metadata:
            return
        
        self.current_timestep = (self.current_timestep - 1) % self.num_timesteps
        self.current_camera_idx = 0  # Reset camera index
        self.load_dataset()
        self.display_current_image()
    
    def change_sequence(self, event=None):
        """Change the current sequence"""
        self.current_sequence = self.sequence_var.get()
        self.current_timestep = 0
        self.current_camera_idx = 0
        self.load_metadata()
        self.load_dataset()
        self.display_current_image()
    
    def update_status(self, message):
        """Update the status bar with a message"""
        self.status_text.set(message)
        print(message)
    
    def on_resize(self, event):
        """Handle window resize events"""
        # Only redraw if we have a valid image
        if hasattr(self, 'photo_image'):
            self.display_current_image()
    
    def create_ui(self):
        """Create the UI elements"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top controls
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Sequence selection
        ttk.Label(top_frame, text="Sequence:").pack(side=tk.LEFT, padx=(0, 5))
        self.sequence_var = tk.StringVar(value=self.current_sequence if self.current_sequence else "")
        sequence_combo = ttk.Combobox(top_frame, textvariable=self.sequence_var, values=self.sequences)
        sequence_combo.pack(side=tk.LEFT, padx=(0, 10))
        sequence_combo.bind("<<ComboboxSelected>>", self.change_sequence)
        
        # Information display
        self.info_text = tk.StringVar(value="No image loaded")
        ttk.Label(top_frame, textvariable=self.info_text).pack(side=tk.LEFT, padx=10)
        
        # Image canvas
        self.image_canvas = tk.Canvas(main_frame, bg="black")
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        self.image_canvas.bind("<Configure>", self.on_resize)
        
        # Bottom navigation controls
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Define a custom style for larger buttons
        style = ttk.Style()
        style.configure("Large.TButton", font=("Arial", 12, "bold"), padding=(10, 5))
        
        # Navigation buttons (larger size)
        ttk.Button(
            bottom_frame, 
            text="← Previous Camera", 
            command=self.prev_camera,
            style="Large.TButton"
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            bottom_frame, 
            text="Next Camera →", 
            command=self.next_camera,
            style="Large.TButton"
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            bottom_frame, 
            text="← Previous Timestep", 
            command=self.prev_timestep,
            style="Large.TButton"
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            bottom_frame, 
            text="Next Timestep →", 
            command=self.next_timestep,
            style="Large.TButton"
        ).pack(side=tk.LEFT, padx=5)
        
        # Image type selection frame
        image_type_frame = ttk.Frame(main_frame)
        image_type_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(image_type_frame, text="Image Type:").pack(side=tk.LEFT, padx=(0, 5))
        
        # Image type buttons
        image_types = [
            ("RGB", "rgb"),
            ("Segmentation", "segmentation"),
            ("Depth", "depth"),
            ("Flow", "flow"),
            ("Prop Depth", "prop_depth"),
            ("FG Mask", "fg_mask"),
            ("Foreground", "foreground"),
            ("Masked Depth", "masked_depth")
        ]
        
        # Create buttons for image types
        for label, type_id in image_types:
            ttk.Button(
                image_type_frame,
                text=label,
                command=lambda t=type_id: self.set_image_type(t),
                style="Large.TButton"
            ).pack(side=tk.LEFT, padx=5)
        
        # Status bar
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)
        
        self.status_text = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_text, anchor=tk.W).pack(side=tk.LEFT)

    def load_model_parameters(self, saved_model_path):
        """Load model parameters from saved file and convert to render variables"""
        try:
            self.update_status(f"Loading model parameters from {saved_model_path}")
            saved_params = dict(np.load(saved_model_path))
            
            # Check if we have the necessary parameters for the current timestep
            if 'means3D' in saved_params and self.current_timestep < len(saved_params['means3D']):
                # Convert the saved parameters to torch tensors
                rendervar_params = {}
                rendervar_params['means3D'] = torch.tensor(saved_params['means3D'][self.current_timestep]).cuda().float()
                rendervar_params['rgb_colors'] = torch.tensor(saved_params['rgb_colors'][self.current_timestep]).cuda().float()
                rendervar_params['unnorm_rotations'] = torch.tensor(saved_params['unnorm_rotations'][self.current_timestep]).cuda().float()
                
                # Check if these are single-timestep parameters
                if len(saved_params['logit_opacities'].shape) == 2:
                    rendervar_params['logit_opacities'] = torch.tensor(saved_params['logit_opacities']).cuda().float()
                else:
                    rendervar_params['logit_opacities'] = torch.tensor(saved_params['logit_opacities'][self.current_timestep]).cuda().float()
                    
                if len(saved_params['log_scales'].shape) == 2:
                    rendervar_params['log_scales'] = torch.tensor(saved_params['log_scales']).cuda().float()
                else:
                    rendervar_params['log_scales'] = torch.tensor(saved_params['log_scales'][self.current_timestep]).cuda().float()
                
                # Convert to rendering variables format
                rendervar = params2rendervar(rendervar_params)
                self.update_status("Successfully loaded parameters for rendering")
                return rendervar
            else:
                self.update_status(f"Parameters for timestep {self.current_timestep} not found in saved model")
                return None
        except Exception as e:
            self.update_status(f"Error loading model parameters: {str(e)}")
            return None

    def show_progress_dialog(self, title="Processing", message="Please wait..."):
        """Show a progress dialog during long operations"""
        progress_window = tk.Toplevel(self.root)
        progress_window.title(title)
        progress_window.geometry("300x100")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        ttk.Label(progress_window, text=message, wraplength=280).pack(pady=10)
        
        progress_bar = ttk.Progressbar(progress_window, mode="indeterminate", length=200)
        progress_bar.pack(pady=10)
        progress_bar.start(10)
        
        progress_window.update()
        return progress_window

def main():
    root = tk.Tk()
    app = ImageViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main()