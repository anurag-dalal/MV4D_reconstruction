import open3d as o3d
import numpy as np

class VoxelGridManager:
    def __init__(self, init_pointcloud_path:str, voxel_size=0.1):
        self.voxel_size = voxel_size
        self.voxel_grid = o3d.geometry.VoxelGrid()
        self.points = np.empty((0, 3))  # Initialize an empty array for points
        self.colors = np.empty((0, 3))  # Initialize an empty array for points
        self.origin = None  # Store the origin of the voxel grid
        if init_pointcloud_path is not None:
            self.initialize_points_from_pointcloud(init_pointcloud_path)
            self.avg, self.min, self.max = self.get_distances()
            self.voxel_size = self.avg * 4
            self._update_voxel_grid()

    def check_if_included_in_voxel(self, points):
        """
        Check if points are included in the voxel grid.
        :param points: numpy array or tensor of shape [N, 3]
        :return: indices of self.points that are in the same voxels as the input points
        """
        if self.voxel_grid is None or len(self.points) == 0:
            return np.array([], dtype=int)
        
        # Convert to numpy if it's a tensor
        if hasattr(points, 'cpu') or hasattr(points, 'cuda'):
            new_points = points.cpu().detach().numpy()
        else:
            new_points = points
        # print('--------------------', type(new_points), new_points.shape)    
        # Create voxels for the input points using the same voxel size
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(new_points)
        input_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=self.voxel_size)
        
        # Get voxel indices for input points
        input_voxels = input_voxel_grid.get_voxels()
        input_voxel_indices = set(tuple(voxel.grid_index) for voxel in input_voxels)
        
        # Get voxel indices for all self.points
        self_pcd = o3d.geometry.PointCloud()
        self_pcd.points = o3d.utility.Vector3dVector(self.points)
        
        # Use the same origin and voxel size for consistency
        min_bound = input_voxel_grid.get_min_bound()
        
        # Find which of self.points fall within the same voxels as input points
        result_indices = []
        for i, point in enumerate(self.points):
            # Calculate the voxel index for this point
            voxel_idx = tuple(np.floor((point - min_bound) / self.voxel_size).astype(int))
            if voxel_idx in input_voxel_indices:
                result_indices.append(i)
        
        return np.array(result_indices, dtype=int)
    
    def initialize_points_from_pointcloud(self, data_path):
        init_pt_cld = np.load(data_path)["data"]
        self.points = init_pt_cld[:, :3]
        self.colors = init_pt_cld[:, 3:6]
        self._update_voxel_grid()
    
    def reinitialize(self, points, colors):
        self.points = points
        self.colors = colors
        self._update_voxel_grid()
    
    def add_points(self, new_points, colors=None):
        """
        Add new points to the voxel grid.
        :param new_points: numpy array of shape [N, 3]
        """
        if colors is not None and colors.shape[0] != new_points.shape[0]:
            raise ValueError("Number of colors must match number of points.")
        if len(new_points) == 0:
            return
        
        
        self.points = np.vstack((self.points, new_points))
        if colors is not None:
            self.colors = np.vstack((self.colors, colors))
        else:
            # If no colors are provided, use a default color (e.g., white)
            default_color = np.array([[0.5, 0.5, 0.5]] * len(new_points))
            self.colors = np.vstack((self.colors, default_color))
        self._update_voxel_grid()

    def remove_points(self, points_to_remove):
        """
        Remove specific points from the voxel grid.
        :param points_to_remove: numpy array of shape [M, 3]
        """
        self.points = np.array([p for p in self.points if not any(np.all(p == r) for r in points_to_remove)])
        self._update_voxel_grid()

    def clear_points(self):
        """
        Clear all points from the voxel grid.
        """
        self.points = np.empty((0, 3))
        self._update_voxel_grid()

    def get_voxel_count(self):
        """
        Get the number of voxels in the voxel grid.
        :return: int
        """
        return len(self.voxel_grid.get_voxels())

    def _update_voxel_grid(self):
        """
        Update the voxel grid based on the current points.
        """
        if len(self.points) > 0:
            point_cloud = o3d.geometry.PointCloud()
            # print('self.points--------------------', type(self.points), self.points.shape)
            point_cloud.points = o3d.utility.Vector3dVector(self.points)
            self.voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size=self.voxel_size)
            # Store the origin of the voxel grid for proper alignment
            voxels = self.voxel_grid.get_voxels()
            if voxels:
                min_indices = np.array([voxel.grid_index for voxel in voxels]).min(axis=0)
                self.origin = min_indices * self.voxel_size
            else:
                self.origin = np.zeros(3)
        else:
            self.voxel_grid = o3d.geometry.VoxelGrid()
            self.origin = None
            
    def visualize(self, point_size=5.0, point_color=None, voxel_color=None):
        """
        Visualize the voxel grid as wireframe and the points.
        
        :param point_size: Size of the points in the visualization
        :param point_color: Color of points as [R, G, B], default is [1, 0, 0] (red)
        :param voxel_color: Color of voxel wireframe as [R, G, B], default is [0, 1, 0] (green)
        """
        if point_color is None:
            point_color = [1, 0, 0]  # Default red for points
        if voxel_color is None:
            voxel_color = [0, 1, 0]  # Default green for voxel wireframe
            
        # Create visualization window
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        # Add point cloud
        if len(self.points) > 0:
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(self.points)
            point_cloud.colors = o3d.utility.Vector3dVector(self.colors)
            # point_cloud.paint_uniform_color(point_color)
            vis.add_geometry(point_cloud)
            
            # Set point size
            render_option = vis.get_render_option()
            render_option.point_size = point_size
        
        # Add voxel grid with wireframe rendering
        if self.get_voxel_count() > 0:
            # Get the actual voxels from Open3D
            voxels = self.voxel_grid.get_voxels()
            
            # Create a list to hold all corner points
            all_corners = []
            all_lines = []
            
            # Get the voxel grid origin directly from Open3D voxel grid
            min_bound = self.voxel_grid.get_min_bound()
            
            # Create wireframe representing the voxels
            for voxel in voxels:
                # Get the actual position in world coordinates
                grid_idx = np.array(voxel.grid_index)
                # Calculate the actual position of this voxel in the same coordinate system as the point cloud
                voxel_position = min_bound + grid_idx * self.voxel_size
                
                x, y, z = voxel_position
                corners = [
                    [x, y, z],
                    [x + self.voxel_size, y, z],
                    [x + self.voxel_size, y + self.voxel_size, z],
                    [x, y + self.voxel_size, z],
                    [x, y, z + self.voxel_size],
                    [x + self.voxel_size, y, z + self.voxel_size],
                    [x + self.voxel_size, y + self.voxel_size, z + self.voxel_size],
                    [x, y + self.voxel_size, z + self.voxel_size]
                ]
                
                # Define lines connecting corners to form a cube
                base_idx = len(all_corners)
                all_corners.extend(corners)
                
                indices = [[0, 1], [1, 2], [2, 3], [3, 0],
                           [4, 5], [5, 6], [6, 7], [7, 4],
                           [0, 4], [1, 5], [2, 6], [3, 7]]
                
                # Add the lines for this voxel with the correct base index
                all_lines.extend([[base_idx + i, base_idx + j] for i, j in indices])
            
            if all_corners:
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(all_corners)
                line_set.lines = o3d.utility.Vector2iVector(all_lines)
                line_set.paint_uniform_color(voxel_color)
                vis.add_geometry(line_set)
            
            # Set voxel rendering to wireframe
            render_option = vis.get_render_option()
            render_option.mesh_show_wireframe = True
        
        # Run the visualizer
        vis.run()
        vis.destroy_window()
    def get_distances(self):
        """
        Calculate the average, minimum, and maximum distances between points.
        :return: avg_distance, min_distance, max_distance or (0, 0, 0) if no points
        """
        if len(self.points) == 0:
            return 0, 0, 0
            
        from sklearn.neighbors import NearestNeighbors
        
        # Use k=4 to get 3 nearest neighbors (first one is the point itself)
        nbrs = NearestNeighbors(n_neighbors=4).fit(self.points)
        distances, _ = nbrs.kneighbors(self.points)
        
        # Skip the first column (distance to self is 0)
        distances = distances[:, 1:]
        
        avg_distance = np.mean(distances)
        min_distance = np.min(distances)
        max_distance = np.max(distances)
        
        return avg_distance, min_distance, max_distance 

# Example usage
if __name__ == "__main__":
    
    sequence = "basketball"
    previous_model_path = f"/home/anurag/Codes/MV4D_reconstruction/output/exp1/{sequence}/params.npz"
    # saved_params = dict(np.load(previous_model_path))
    # points = saved_params['means3D'][0]
    # points = points.reshape(-1, 3)
    # manager.add_points(points)
    # print("Initial voxel count:", manager.get_voxel_count())
    # manager.visualize(point_size=5.0, point_color=[0, 0, 1], voxel_color=[1, 0, 0])
    from datamanager import datamanager
    data_manager = datamanager(previous_model_path, is_numpy=True)
    data_manager.read_data_at_timestep(0)
    points = data_manager.get_data()['means3D'].cpu().numpy()
    colors = data_manager.get_data()['rgb_colors'].cpu().numpy()
    avg, min, max = data_manager.get_distances()
    manager = VoxelGridManager(voxel_size=avg*4)
    manager.add_points(points, colors)
    manager.visualize(point_size=min, point_color=[0, 0, 1], voxel_color=[1, 0, 0])
    


    


    # # Add points
    # points = np.array([[0, 0, 0], [0.1, 0.1, 0.1], [1, 1, 1]])
    # manager.add_points(points)
    # print("Voxel count after adding points:", manager.get_voxel_count())

    # # Remove a point
    # manager.remove_points(np.array([[0, 0, 0]]))
    # print("Voxel count after removing a point:", manager.get_voxel_count())
    
    # # Visualize the voxel grid and points
    # manager.visualize(point_size=10.0)

    # # Clear all points
    # manager.clear_points()
    # print("Voxel count after clearing points:", manager.get_voxel_count())