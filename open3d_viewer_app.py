import tkinter as tk
from tkinter import ttk
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import time
from PIL import Image, ImageTk
import os
import tempfile
import sys

class Open3DViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Open3D Viewer with Matplotlib")
        self.geometry("1920x1080")
        
        # Create main frame
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create left frame for Open3D visualization
        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create right frame for Matplotlib plots
        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Setup Open3D visualization widget
        self.setup_open3d_widget()
        
        # Setup Matplotlib widget
        self.setup_matplotlib_widget()
        
        # Add control buttons
        self.setup_control_buttons()
        
        # Create a simple point cloud
        self.pointcloud = self.create_demo_pointcloud()
        
        # Create coordinate frame for reference
        self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0]
        )
        
        # Store geometries
        self.geometries = [self.pointcloud, self.coordinate_frame]
        
        # Initial camera parameters (will be updated after first render)
        self.view_params = {
            'lookat': [0.0, 0.0, 0.0],
            'up': [0.0, 1.0, 0.0],
            'front': [0.0, 0.0, -1.0],
            'zoom': 0.7
        }
        
        # Initial rendering
        self.render_open3d()
        
        # Update matplotlib plot initially
        self.update_plot()
        
        # Protocol for closing the window properly
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_open3d_widget(self):
        # Create a frame for the Open3D rendering
        self.o3d_frame = ttk.LabelFrame(self.left_frame, text="Open3D Visualization")
        self.o3d_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a canvas to display the rendered image
        self.o3d_canvas = tk.Canvas(self.o3d_frame, bg='black')
        self.o3d_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Add mouse event bindings to rotate/move the view
        self.o3d_canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.o3d_canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.o3d_canvas.bind("<MouseWheel>", self.on_mouse_wheel)  # Windows
        self.o3d_canvas.bind("<Button-4>", self.on_mouse_wheel)    # Linux scroll up
        self.o3d_canvas.bind("<Button-5>", self.on_mouse_wheel)    # Linux scroll down
        
        # Store last mouse position
        self.last_x = 0
        self.last_y = 0

    def on_mouse_down(self, event):
        self.last_x = event.x
        self.last_y = event.y

    def on_mouse_move(self, event):
        dx = event.x - self.last_x
        dy = event.y - self.last_y
        
        # Rotate the view
        if dx != 0 or dy != 0:
            # Convert to spherical coordinates
            r = np.linalg.norm(self.view_params['front'])
            theta = np.arccos(self.view_params['front'][1] / r)
            phi = np.arctan2(self.view_params['front'][2], self.view_params['front'][0])
            
            # Update angles
            phi -= dx * 0.01
            theta -= dy * 0.01
            theta = np.clip(theta, 0.001, np.pi - 0.001)
            
            # Convert back to Cartesian
            self.view_params['front'][0] = r * np.sin(theta) * np.cos(phi)
            self.view_params['front'][1] = r * np.cos(theta)
            self.view_params['front'][2] = r * np.sin(theta) * np.sin(phi)
            
            # Render the updated view
            self.render_open3d()
            
        self.last_x = event.x
        self.last_y = event.y

    def on_mouse_wheel(self, event):
        # Get scroll direction
        if event.num == 4 or (hasattr(event, 'delta') and event.delta > 0):
            # Zoom in
            self.view_params['zoom'] *= 1.1
        else:
            # Zoom out
            self.view_params['zoom'] /= 1.1
        
        # Render the updated view
        self.render_open3d()

    def setup_matplotlib_widget(self):
        # Create a frame for the Matplotlib plot
        self.plot_frame = ttk.LabelFrame(self.right_frame, text="Matplotlib Plot")
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a Figure for the plot
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.plot = self.figure.add_subplot(111)
        
        # Create a canvas to display the plot
        self.canvas = FigureCanvasTkAgg(self.figure, self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def setup_control_buttons(self):
        # Create a frame for the control buttons
        self.control_frame = ttk.Frame(self.right_frame)
        self.control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Add a button to update the plot
        self.update_button = ttk.Button(
            self.control_frame, 
            text="Update Plot", 
            command=self.update_plot
        )
        self.update_button.pack(side=tk.LEFT, padx=5)
        
        # Add a button to rotate the point cloud
        self.rotate_button = ttk.Button(
            self.control_frame, 
            text="Rotate Point Cloud", 
            command=self.rotate_pointcloud
        )
        self.rotate_button.pack(side=tk.LEFT, padx=5)
        
        # Add a button to reset the view
        self.reset_button = ttk.Button(
            self.control_frame, 
            text="Reset View", 
            command=self.reset_view
        )
        self.reset_button.pack(side=tk.LEFT, padx=5)

    def create_demo_pointcloud(self):
        # Create a simple point cloud with random colors
        # Make it more visually interesting with a pattern
        t = np.linspace(0, 4 * np.pi, 2000)
        x = t * np.cos(t) * 0.1
        y = np.sin(t) * 0.1
        z = t * np.sin(t) * 0.1
        
        points = np.vstack([x, y, z]).T
        
        # Create gradient colors
        colors = np.zeros_like(points)
        colors[:, 0] = (points[:, 0] - points[:, 0].min()) / (points[:, 0].max() - points[:, 0].min())  # R
        colors[:, 1] = (points[:, 1] - points[:, 1].min()) / (points[:, 1].max() - points[:, 1].min())  # G
        colors[:, 2] = (points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min())  # B
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd

    def render_open3d(self):
        # Create a new visualizer each time to avoid OpenGL context issues
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=800, height=600)
        
        # Add the geometries
        for geometry in self.geometries:
            vis.add_geometry(geometry)
        
        # Set render options
        render_option = vis.get_render_option()
        render_option.background_color = np.array([0.1, 0.1, 0.1])
        render_option.point_size = 2.0
        
        # Apply view parameters
        ctr = vis.get_view_control()
        ctr.set_lookat(self.view_params['lookat'])
        ctr.set_up(self.view_params['up'])
        ctr.set_front(self.view_params['front'])
        ctr.set_zoom(self.view_params['zoom'])
        
        # Render once to update view
        vis.poll_events()
        vis.update_renderer()
        
        # Capture image
        img = vis.capture_screen_float_buffer(True)
        
        # Save current view parameters
        param = ctr.convert_to_pinhole_camera_parameters()
        # We could save and restore the camera parameters if needed
        
        # Close visualizer to release OpenGL context
        vis.destroy_window()
        
        # Convert to numpy array and then to tkinter PhotoImage
        img_np = (np.asarray(img) * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        
        # Resize to fit canvas if needed
        canvas_width = self.o3d_canvas.winfo_width()
        canvas_height = self.o3d_canvas.winfo_height()
        if canvas_width > 1 and canvas_height > 1:  # Ensure canvas has been drawn
            img_pil = img_pil.resize((canvas_width, canvas_height))
        
        # Convert to Tkinter PhotoImage and display
        self.img_tk = ImageTk.PhotoImage(image=img_pil)
        self.o3d_canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)

    def update_plot(self):
        # Clear the previous plot
        self.plot.clear()
        
        # Generate some random data for a demo graph
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + np.random.random(100) * 0.2
        
        # Plot the data
        self.plot.plot(x, y, 'b-')
        self.plot.set_title('Demo Graph')
        self.plot.set_xlabel('X-axis')
        self.plot.set_ylabel('Y-axis')
        self.plot.grid(True)
        
        # Update the canvas
        self.canvas.draw()

    def rotate_pointcloud(self):
        # Rotate the point cloud around the y-axis
        rotation = o3d.geometry.get_rotation_matrix_from_xyz([0, 0.1, 0])
        self.pointcloud.rotate(rotation)
        
        # Update the visualization
        self.render_open3d()

    def reset_view(self):
        # Reset camera view parameters
        self.view_params = {
            'lookat': [0.0, 0.0, 0.0],
            'up': [0.0, 1.0, 0.0],
            'front': [0.0, 0.0, -1.0],
            'zoom': 0.7
        }
        
        # Update the visualization
        self.render_open3d()

    def on_closing(self):
        # Destroy the Tkinter window
        self.destroy()


if __name__ == "__main__":
    app = Open3DViewer()
    app.mainloop()