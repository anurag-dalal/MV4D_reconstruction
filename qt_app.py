import sys
import numpy as np
import open3d as o3d
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLabel, QFrame
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import threading
import time


class MplCanvas(FigureCanvas):
    """Matplotlib canvas class for embedding plots in Qt"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class Open3DQtVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Open3D with Matplotlib in Qt")
        self.setGeometry(100, 100, 1600, 900)
        
        # Create the main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        
        # Create left panel for Open3D visualization
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        
        # Create label for displaying Open3D render
        self.vis_label = QLabel()
        self.vis_label.setMinimumSize(800, 600)
        self.vis_label.setAlignment(Qt.AlignCenter)
        self.vis_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.vis_label.setStyleSheet("background-color: #333333;")
        self.left_layout.addWidget(self.vis_label)
        
        # Create buttons for Open3D controls
        self.buttons_layout = QHBoxLayout()
        self.rotate_button = QPushButton("Rotate Point Cloud")
        self.rotate_button.clicked.connect(self.rotate_pointcloud)
        self.reset_view_button = QPushButton("Reset View")
        self.reset_view_button.clicked.connect(self.reset_view)
        self.buttons_layout.addWidget(self.rotate_button)
        self.buttons_layout.addWidget(self.reset_view_button)
        self.left_layout.addLayout(self.buttons_layout)
        
        # Create right panel for Matplotlib visualization
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        
        # Create the matplotlib canvas
        self.plot_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.right_layout.addWidget(self.plot_canvas)
        
        # Create button to update the plot
        self.update_plot_button = QPushButton("Update Plot")
        self.update_plot_button.clicked.connect(self.update_plot)
        self.right_layout.addWidget(self.update_plot_button)
        
        # Add panels to main layout
        self.main_layout.addWidget(self.left_panel)
        self.main_layout.addWidget(self.right_panel)
        
        # Create a simple point cloud
        self.pointcloud = self.create_demo_pointcloud()
        
        # Create coordinate frame for reference
        self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0, origin=[0, 0, 0]  # Increased size from 0.5 to 1.0
        )
        
        # Store geometries
        self.geometries = [self.pointcloud, self.coordinate_frame]
        
        # Initial camera parameters - adjusted for better visibility
        self.view_params = {
            'lookat': [0.0, 0.0, 0.0],  # Look at center
            'up': [0.0, 1.0, 0.0],      # Y-axis up
            'front': [0.0, 0.0, -1.0],  # Looking from z-negative direction
            'zoom': 0.5                 # Zoomed out more to see the whole point cloud
        }
        
        # Set up a timer to render Open3D periodically
        self.timer = QTimer()
        self.timer.timeout.connect(self.render_open3d)
        self.timer.start(100)  # Update every 100ms
        
        # Initial rendering and plot
        self.render_open3d()
        self.update_plot()
        
    def create_demo_pointcloud(self):
        """Create a demo point cloud with a spiral pattern and gradient colors"""
        # Create more points with larger scale for better visibility
        t = np.linspace(0, 8 * np.pi, 5000)  # More points, more turns
        x = t * np.cos(t) * 0.3  # Increased scale from 0.1 to 0.3
        y = np.sin(t) * 0.3      # Increased scale
        z = t * np.sin(t) * 0.3  # Increased scale
        
        points = np.vstack([x, y, z]).T
        
        # Create gradient colors with more vibrant colors
        colors = np.zeros_like(points)
        colors[:, 0] = (points[:, 0] - points[:, 0].min()) / (points[:, 0].max() - points[:, 0].min())  # R
        colors[:, 1] = (points[:, 1] - points[:, 1].min()) / (points[:, 1].max() - points[:, 1].min())  # G
        colors[:, 2] = (points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min())  # B
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Print debug info
        print(f"Created point cloud with {len(points)} points")
        print(f"Point cloud bounds: X[{min(x):.2f}, {max(x):.2f}], Y[{min(y):.2f}, {max(y):.2f}], Z[{min(z):.2f}, {max(z):.2f}]")
        
        return pcd
        
    def render_open3d(self):
        """Render Open3D visualization and display it in the Qt window"""
        # Create a new visualizer each time
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=800, height=600)
        
        # Add the geometries
        for geometry in self.geometries:
            vis.add_geometry(geometry)
        
        # Set render options
        render_option = vis.get_render_option()
        render_option.background_color = np.array([0.1, 0.1, 0.1])
        render_option.point_size = 5.0  # Increased from 2.0 to 5.0 for better visibility
        render_option.show_coordinate_frame = True  # Explicitly enable coordinate frame
        
        # Apply view parameters
        ctr = vis.get_view_control()
        ctr.set_lookat(self.view_params['lookat'])
        ctr.set_up(self.view_params['up'])
        ctr.set_front(self.view_params['front'])
        ctr.set_zoom(self.view_params['zoom'])
        
        # Render once
        vis.poll_events()
        vis.update_renderer()
        
        # Capture image
        img = vis.capture_screen_float_buffer(True)
        # Close visualizer
        vis.destroy_window()
        
        # Convert to numpy array and then to Qt QImage/QPixmap
        img_np = (np.asarray(img) * 255).astype(np.uint8)
        import cv2
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite("screenshot.png", img_np)  # Save screenshot for debugging
        print(f"Captured image shape: {img_np.shape}")
        h, w, c = img_np.shape
        bytes_per_line = 3 * w
        q_img = QImage(img_np.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        
        # Resize to fit label if needed
        label_size = self.vis_label.size()
        if label_size.width() > 0 and label_size.height() > 0:
            pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
        # Set the pixmap to the label
        self.vis_label.setPixmap(pixmap)
        
    def update_plot(self):
        """Update the matplotlib plot with new data"""
        # Clear the current plot
        self.plot_canvas.axes.clear()
        
        # Generate new random data
        x = np.linspace(0, 10, 100)
        y = np.sin(x) + np.random.random(100) * 0.2
        
        # Create the plot
        self.plot_canvas.axes.plot(x, y, 'b-')
        self.plot_canvas.axes.set_title('Demo Plot')
        self.plot_canvas.axes.set_xlabel('X-axis')
        self.plot_canvas.axes.set_ylabel('Y-axis')
        self.plot_canvas.axes.grid(True)
        
        # Refresh canvas
        self.plot_canvas.draw()
        
    def rotate_pointcloud(self):
        """Rotate the point cloud around the y-axis"""
        rotation = o3d.geometry.get_rotation_matrix_from_xyz([0, 0.1, 0])
        self.pointcloud.rotate(rotation)
        self.render_open3d()
        
    def reset_view(self):
        """Reset camera view parameters"""
        self.view_params = {
            'lookat': [0.0, 0.0, 0.0],
            'up': [0.0, 1.0, 0.0],
            'front': [0.0, 0.0, -1.0],
            'zoom': 0.5  # Adjusted zoom for better visibility
        }
        self.render_open3d()
        
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
                    
                    # Render the updated view
                    self.render_open3d()
            
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
            
        # Render the updated view
        self.render_open3d()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Open3DQtVisualizer()
    window.show()
    sys.exit(app.exec_())