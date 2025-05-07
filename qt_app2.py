import sys
import os
import numpy as np
import open3d as o3d
import torch
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QWidget, QPushButton, QLabel, QFrame, QGroupBox,
                            QComboBox, QTextEdit, QSplitter, QCheckBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont

# Import functions from visualize.py
from visualize import (load_scene_data, render, rgbd2pcd, init_camera,
                       calculate_trajectories, calculate_rot_vec)

# Constants from visualize.py
w, h = 640, 360
near, far = 0.01, 100.0
view_scale = 3.9
fps = 20


class Open3DGaussianVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize UI state
        self.ui_state = {
            'is_playing': False,
            'current_frame': 0,
            'render_mode': 'color',  # 'color', 'depth', or 'centers'
            'additional_lines': None,  # None, 'trajectories', or 'rotations'
            'remove_background': False,
            'force_loop': False,
            'last_time': 0,
            'need_redraw': True,
        }
        
        # Scene data
        self.scene_data = None
        self.is_fg = None
        self.num_timesteps = 0
        self.current_sequence = "basketball"
        self.current_experiment = "exp1"
        
        # Camera parameters
        self.w2c = None
        self.k = None
        
        # Visualization objects
        self.pcd = None
        self.lines = None
        self.linesets = None
        
        # Last mouse position for camera control
        self.last_mouse_pos = None
        
        self.init_ui()
        self.load_data(self.current_sequence, self.current_experiment)
        
    def init_ui(self):
        self.setWindowTitle("Gaussian Visualizer with Open3D")
        self.setGeometry(100, 100, 1600, 900)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create left panel for Open3D visualization
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Create label for displaying Open3D render
        self.vis_label = QLabel()
        self.vis_label.setMinimumSize(800, 600)
        self.vis_label.setAlignment(Qt.AlignCenter)
        self.vis_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.vis_label.setStyleSheet("background-color: #333333;")
        left_layout.addWidget(self.vis_label)
        
        # Add playback controls
        playback_controls = QHBoxLayout()
        
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)
        
        self.prev_frame_button = QPushButton("< Prev")
        self.prev_frame_button.clicked.connect(self.prev_frame)
        
        self.next_frame_button = QPushButton("Next >")
        self.next_frame_button.clicked.connect(self.next_frame)
        
        self.reset_view_button = QPushButton("Reset View")
        self.reset_view_button.clicked.connect(self.reset_view)
        
        playback_controls.addWidget(self.prev_frame_button)
        playback_controls.addWidget(self.play_button)
        playback_controls.addWidget(self.next_frame_button)
        playback_controls.addWidget(self.reset_view_button)
        
        left_layout.addLayout(playback_controls)
        
        # Create right panel with controls and camera params
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Create a group box for rendering options
        render_group = QGroupBox("Rendering Options")
        render_layout = QVBoxLayout(render_group)
        
        # Render mode selection
        render_mode_layout = QHBoxLayout()
        render_mode_label = QLabel("Render Mode:")
        self.render_mode_combo = QComboBox()
        self.render_mode_combo.addItems(["Color", "Depth", "Centers"])
        self.render_mode_combo.currentIndexChanged.connect(self.change_render_mode)
        render_mode_layout.addWidget(render_mode_label)
        render_mode_layout.addWidget(self.render_mode_combo)
        render_layout.addLayout(render_mode_layout)
        
        # Background toggle
        self.bg_checkbox = QCheckBox("Remove Background")
        self.bg_checkbox.setChecked(False)
        self.bg_checkbox.toggled.connect(self.toggle_background)
        render_layout.addWidget(self.bg_checkbox)
        
        # Visualization lines
        lines_layout = QHBoxLayout()
        lines_label = QLabel("Show Lines:")
        self.lines_combo = QComboBox()
        self.lines_combo.addItems(["None", "Trajectories", "Rotations"])
        self.lines_combo.currentIndexChanged.connect(self.change_lines_mode)
        lines_layout.addWidget(lines_label)
        lines_layout.addWidget(self.lines_combo)
        render_layout.addLayout(lines_layout)
        
        # Dataset selection
        dataset_layout = QHBoxLayout()
        dataset_label = QLabel("Sequence:")
        self.sequence_combo = QComboBox()
        self.sequence_combo.addItems(["basketball", "boxes", "football", "juggle", "softball", "tennis"])
        self.sequence_combo.currentIndexChanged.connect(self.change_sequence)
        dataset_layout.addWidget(dataset_label)
        dataset_layout.addWidget(self.sequence_combo)
        render_layout.addLayout(dataset_layout)
        
        right_layout.addWidget(render_group)
        
        # Camera parameters display
        camera_group = QGroupBox("Camera Parameters")
        camera_layout = QVBoxLayout(camera_group)
        
        self.camera_text = QTextEdit()
        self.camera_text.setReadOnly(True)
        self.camera_text.setFont(QFont("Courier", 10))
        self.camera_text.setStyleSheet("background-color: #f0f0f0;")
        self.camera_text.setMaximumHeight(300)
        camera_layout.addWidget(self.camera_text)
        
        right_layout.addWidget(camera_group)
        
        # Add a status console
        status_group = QGroupBox("Status Console")
        status_layout = QVBoxLayout(status_group)
        
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setFont(QFont("Courier", 10))
        self.status_text.setStyleSheet("background-color: #f0f0f0;")
        status_layout.addWidget(self.status_text)
        
        right_layout.addWidget(status_group)
        
        # Add panels to main layout
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([1200, 400])  # Initial sizes
        main_layout.addWidget(splitter)
        
        # Set up timer for rendering
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_view)
        self.timer.start(50)  # 20 FPS
        
        # Log initial status
        self.log_status("Application initialized. Ready to visualize.")
        
    def log_status(self, message):
        """Add a message to the status console"""
        self.status_text.append(f"{time.strftime('%H:%M:%S')}: {message}")
        # Scroll to bottom
        scroll_bar = self.status_text.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())
        
    def update_camera_display(self):
        """Update the camera parameters display"""
        if self.w2c is not None and self.k is not None:
            text = "Camera Extrinsic (world to camera):\n"
            for row in self.w2c:
                text += f"{row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f} {row[3]:8.4f}\n"
            
            text += "\nCamera Intrinsic K:\n"
            for row in self.k:
                text += f"{row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f}\n"
                
            # Add view parameters
            text += "\nView Parameters:\n"
            text += f"Look-at: [{self.view_params['lookat'][0]:.4f}, {self.view_params['lookat'][1]:.4f}, {self.view_params['lookat'][2]:.4f}]\n"
            text += f"Up: [{self.view_params['up'][0]:.4f}, {self.view_params['up'][1]:.4f}, {self.view_params['up'][2]:.4f}]\n"
            text += f"Front: [{self.view_params['front'][0]:.4f}, {self.view_params['front'][1]:.4f}, {self.view_params['front'][2]:.4f}]\n"
            text += f"Zoom: {self.view_params['zoom']:.4f}"
            
            self.camera_text.setText(text)
            
    def load_data(self, sequence, experiment):
        """Load scene data for a given sequence and experiment"""
        try:
            self.log_status(f"Loading data for sequence '{sequence}' from experiment '{experiment}'...")
            
            # Load scene data
            self.scene_data, self.is_fg = load_scene_data(sequence, experiment)
            self.num_timesteps = len(self.scene_data)
            
            # Initialize camera
            self.w2c, self.k = init_camera()
            
            # Initialize view parameters
            self.view_params = {
                'lookat': [0.0, 0.0, 0.0],
                'up': [0.0, 1.0, 0.0],
                'front': [0.0, 0.0, -1.0],
                'zoom': 0.5
            }
            
            # Initialize point cloud
            self.pcd = o3d.geometry.PointCloud()
            
            # Render first frame
            self.ui_state['current_frame'] = 0
            self.ui_state['need_redraw'] = True
            self.update_view()
            
            self.log_status(f"Loaded {self.num_timesteps} frames successfully.")
            
        except Exception as e:
            self.log_status(f"Error loading data: {str(e)}")
            
    def update_view(self):
        """Update the Open3D visualization based on current state"""
        if self.scene_data is None or not self.ui_state['need_redraw']:
            return
            
        try:
            # Get current frame index
            t = self.ui_state['current_frame'] % self.num_timesteps
            
            # Create visualizer
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False, width=800, height=600)
            
            # Render based on render mode
            if self.ui_state['render_mode'] == 'centers':
                pts = o3d.utility.Vector3dVector(self.scene_data[t]['means3D'].contiguous().double().cpu().numpy())
                cols = o3d.utility.Vector3dVector(self.scene_data[t]['colors_precomp'].contiguous().double().cpu().numpy())
                self.pcd.points = pts
                self.pcd.colors = cols
            else:
                # Create a copy of the timestep data
                rendervar = self.scene_data[t].copy()
                
                # Apply background removal if enabled
                if self.ui_state['remove_background']:
                    # Get the foreground mask
                    is_fg = rendervar['colors_precomp'][:, 0] > 0.5
                    # Set opacity to 0 for background points
                    mask_opacities = rendervar['opacities'].clone()
                    mask_opacities[~is_fg] = 0.0
                    rendervar['opacities'] = mask_opacities
                
                # Render the frame
                im, depth = render(self.w2c, self.k, rendervar)
                
                # Convert to point cloud
                try:
                    pts, cols = rgbd2pcd(im, depth, self.w2c, self.k, show_depth=(self.ui_state['render_mode'] == 'depth'))
                    self.pcd.points = pts
                    self.pcd.colors = cols
                except Exception as e:
                    self.log_status(f"Error in point cloud conversion: {str(e)}")
                    # Fallback to centers visualization if rgbd2pcd fails
                    pts = o3d.utility.Vector3dVector(self.scene_data[t]['means3D'].contiguous().double().cpu().numpy())
                    cols = o3d.utility.Vector3dVector(self.scene_data[t]['colors_precomp'].contiguous().double().cpu().numpy())
                    self.pcd.points = pts
                    self.pcd.colors = cols
            
            vis.add_geometry(self.pcd)
            
            # Add trajectory/rotation lines if enabled
            if self.ui_state['additional_lines'] is not None:
                if self.linesets is None:
                    if self.ui_state['additional_lines'] == 'trajectories':
                        self.linesets = calculate_trajectories(self.scene_data, self.is_fg)
                    else:  # 'rotations'
                        self.linesets = calculate_rot_vec(self.scene_data, self.is_fg)
                
                # Create line set for current frame
                if self.lines is None:
                    self.lines = o3d.geometry.LineSet()
                
                # Get appropriate frame index for line visualization
                if self.ui_state['additional_lines'] == 'trajectories':
                    lt = max(0, min(t - 15, len(self.linesets) - 1))  # traj_length is 15
                else:
                    lt = t
                
                if lt >= 0 and lt < len(self.linesets):
                    self.lines.points = self.linesets[lt].points
                    self.lines.colors = self.linesets[lt].colors
                    self.lines.lines = self.linesets[lt].lines
                    vis.add_geometry(self.lines)
            
            # Set render options
            render_option = vis.get_render_option()
            render_option.background_color = np.array([0.1, 0.1, 0.1])
            render_option.point_size = 5.0
            render_option.show_coordinate_frame = True
            
            # Apply view parameters
            ctr = vis.get_view_control()
            ctr.set_lookat(self.view_params['lookat'])
            ctr.set_up(self.view_params['up'])
            ctr.set_front(self.view_params['front'])
            ctr.set_zoom(self.view_params['zoom'])
            
            # Render
            vis.poll_events()
            vis.update_renderer()
            
            # Capture image
            img = vis.capture_screen_float_buffer(True)
            vis.destroy_window()
            
            # Convert to QPixmap and display
            img_np = (np.asarray(img) * 255).astype(np.uint8)
            
            # Make sure the image is correctly formatted (RGB)
            if img_np.shape[2] == 3:
                h, w, c = img_np.shape
                bytes_per_line = 3 * w
                q_img = QImage(img_np.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                
                # Resize to fit label if needed
                label_size = self.vis_label.size()
                if label_size.width() > 0 and label_size.height() > 0:
                    pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    
                self.vis_label.setPixmap(pixmap)
            else:
                self.log_status(f"Warning: Image has unexpected shape: {img_np.shape}")
            
            # Update camera parameters display
            self.update_camera_display()
            
            # Reset need_redraw flag
            self.ui_state['need_redraw'] = False
            
        except Exception as e:
            import traceback
            self.log_status(f"Error updating view: {str(e)}")
            self.log_status(traceback.format_exc())
    
    # UI Control methods
    def toggle_play(self):
        """Toggle play/pause state"""
        self.ui_state['is_playing'] = not self.ui_state['is_playing']
        self.play_button.setText("Pause" if self.ui_state['is_playing'] else "Play")
        self.log_status(f"Playback {'started' if self.ui_state['is_playing'] else 'paused'}")
        
        # Start animation timer if playing
        if self.ui_state['is_playing']:
            self.animation_timer = QTimer()
            self.animation_timer.timeout.connect(self.advance_frame)
            self.animation_timer.start(1000 // fps)
        else:
            if hasattr(self, 'animation_timer'):
                self.animation_timer.stop()
    
    def advance_frame(self):
        """Advance to the next frame during playback"""
        if self.ui_state['is_playing']:
            self.ui_state['current_frame'] = (self.ui_state['current_frame'] + 1) % self.num_timesteps
            self.ui_state['need_redraw'] = True
    
    def next_frame(self):
        """Go to the next frame"""
        if not self.ui_state['is_playing']:
            self.ui_state['current_frame'] = (self.ui_state['current_frame'] + 1) % self.num_timesteps
            self.ui_state['need_redraw'] = True
            self.log_status(f"Frame: {self.ui_state['current_frame']}/{self.num_timesteps-1}")
    
    def prev_frame(self):
        """Go to the previous frame"""
        if not self.ui_state['is_playing']:
            self.ui_state['current_frame'] = (self.ui_state['current_frame'] - 1) % self.num_timesteps
            self.ui_state['need_redraw'] = True
            self.log_status(f"Frame: {self.ui_state['current_frame']}/{self.num_timesteps-1}")
    
    def reset_view(self):
        """Reset the camera view"""
        self.view_params = {
            'lookat': [0.0, 0.0, 0.0],
            'up': [0.0, 1.0, 0.0],
            'front': [0.0, 0.0, -1.0],
            'zoom': 0.5
        }
        self.ui_state['need_redraw'] = True
        self.log_status("View reset to default")
    
    def change_render_mode(self, index):
        """Change the rendering mode based on combobox selection"""
        modes = ['color', 'depth', 'centers']
        self.ui_state['render_mode'] = modes[index]
        self.ui_state['need_redraw'] = True
        self.log_status(f"Render mode changed to {modes[index]}")
        
        # Clear linesets when changing render mode
        self.linesets = None
    
    def toggle_background(self, checked):
        """Toggle background visibility"""
        self.ui_state['remove_background'] = checked
        self.ui_state['need_redraw'] = True
        self.log_status(f"Background removal {'enabled' if checked else 'disabled'}")
    
    def change_lines_mode(self, index):
        """Change the additional lines visualization mode"""
        modes = [None, 'trajectories', 'rotations']
        self.ui_state['additional_lines'] = modes[index]
        self.linesets = None  # Clear cached linesets
        self.ui_state['need_redraw'] = True
        mode_str = "None" if modes[index] is None else modes[index]
        self.log_status(f"Lines visualization mode changed to {mode_str}")
    
    def change_sequence(self, index):
        """Change the current sequence"""
        sequences = ["basketball", "boxes", "football", "juggle", "softball", "tennis"]
        self.current_sequence = sequences[index]
        self.load_data(self.current_sequence, self.current_experiment)
    
    # Mouse event handlers for camera control
    def mousePressEvent(self, event):
        """Handle mouse press for camera manipulation"""
        self.last_mouse_pos = event.pos()
    
    def mouseMoveEvent(self, event):
        """Handle mouse movement for camera manipulation"""
        if hasattr(self, 'last_mouse_pos'):
            dx = event.x() - self.last_mouse_pos.x()
            dy = event.y() - self.last_mouse_pos.y()
            
            if event.buttons() & Qt.LeftButton:
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
                    
                    self.ui_state['need_redraw'] = True
            
            self.last_mouse_pos = event.pos()
    
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        delta = event.angleDelta().y()
        
        if delta > 0:
            # Zoom in
            self.view_params['zoom'] *= 1.1
        else:
            # Zoom out
            self.view_params['zoom'] /= 1.1
            
        self.ui_state['need_redraw'] = True


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Open3DGaussianVisualizer()
    window.show()
    sys.exit(app.exec_())