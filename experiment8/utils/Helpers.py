import numpy as np
import torch
from diff_gaussian_rasterization_with_depth import GaussianRasterizationSettings as Camera

def setup_camera(w, h, k, w2c, near=0.01, far=100):
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False
    )
    return cam

def visulaize_point_cloud_6d_torch_array(point_cloud_data):
    """
    Visualize a point cloud using Open3D.
    Args:
        point_cloud_data (torch.Tensor): The point cloud data as a tensor of shape (N, 6).
            The first three columns are the x, y, z coordinates, and the last three columns are the RGB colors.
    """
    import open3d as o3d
    import numpy as np

    # Convert the point cloud data to a numpy array
    point_cloud_data = point_cloud_data.cpu().numpy()
    
    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    
    # Set the points and colors
    pcd.points = o3d.utility.Vector3dVector(point_cloud_data[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(point_cloud_data[:, 3:])
    
    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])