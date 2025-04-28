import torch
import numpy as np
import open3d as o3d
import time
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import setup_camera, quat_mult
from external import build_rotation
from colormap import colormap
from copy import deepcopy

# Initial render settings (can be changed via UI)
RENDER_MODE = 'color'  # 'color', 'depth' or 'centers'
ADDITIONAL_LINES = None  # None, 'trajectories' or 'rotations'
REMOVE_BACKGROUND = False  # False or True
FORCE_LOOP = False  # False or True

# Constants
w, h = 640, 360
near, far = 0.01, 100.0
view_scale = 3.9
fps = 20
traj_frac = 25  # 4% of points
traj_length = 15
def_pix = torch.tensor(
    np.stack(np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5, 1), -1).reshape(-1, 3)).cuda().float()
pix_ones = torch.ones(h * w, 1).cuda().float()

# UI control state
ui_state = {
    'is_playing': True,
    'current_frame': 0,
    'render_mode': RENDER_MODE,
    'additional_lines': ADDITIONAL_LINES,
    'remove_background': REMOVE_BACKGROUND,
    'force_loop': FORCE_LOOP,
    'last_time': 0,
    'need_redraw': True,
}

# UI callback definitions
def toggle_play_callback(vis):
    ui_state['is_playing'] = not ui_state['is_playing']
    return False

def next_frame_callback(vis):
    if not ui_state['is_playing']:
        ui_state['current_frame'] += 1
        ui_state['need_redraw'] = True
    return False

def prev_frame_callback(vis):
    if not ui_state['is_playing']:
        ui_state['current_frame'] -= 1
        ui_state['need_redraw'] = True
    return False

def toggle_background_callback(vis):
    ui_state['remove_background'] = not ui_state['remove_background']
    ui_state['need_redraw'] = True
    return False

def change_render_mode_callback(vis, mode):
    ui_state['render_mode'] = mode
    ui_state['need_redraw'] = True
    return False

def toggle_lines_callback(vis, lines_mode):
    if ui_state['additional_lines'] == lines_mode:
        ui_state['additional_lines'] = None
    else:
        ui_state['additional_lines'] = lines_mode
    ui_state['need_redraw'] = True
    return False


def init_camera(y_angle=0., center_dist=2.4, cam_height=1.3, f_ratio=0.82):
    ry = y_angle * np.pi / 180
    w2c = np.array([[np.cos(ry), 0., -np.sin(ry), 0.],
                    [0.,         1., 0.,          cam_height],
                    [np.sin(ry), 0., np.cos(ry),  center_dist],
                    [0.,         0., 0.,          1.]])
    k = np.array([[f_ratio * w, 0, w / 2], [0, f_ratio * w, h / 2], [0, 0, 1]])
    return w2c, k


def load_scene_data(seq, exp, seg_as_col=False):
    params = dict(np.load(f"./output/{exp}/{seq}/params.npz"))
    params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}
    is_fg = params['seg_colors'][:, 0] > 0.5
    scene_data = []
    for t in range(len(params['means3D'])):
        rendervar = {
            'means3D': params['means3D'][t],
            'colors_precomp': params['rgb_colors'][t] if not seg_as_col else params['seg_colors'],
            'rotations': torch.nn.functional.normalize(params['unnorm_rotations'][t]),
            'opacities': torch.sigmoid(params['logit_opacities']),
            'scales': torch.exp(params['log_scales']),
            'means2D': torch.zeros_like(params['means3D'][0], device="cuda")
        }
        scene_data.append(rendervar)
    return scene_data, is_fg


def make_lineset(all_pts, cols, num_lines):
    linesets = []
    for pts in all_pts:
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
        lineset.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(cols, np.float64))
        pt_indices = np.arange(len(lineset.points))
        line_indices = np.stack((pt_indices, pt_indices - num_lines), -1)[num_lines:]
        lineset.lines = o3d.utility.Vector2iVector(np.ascontiguousarray(line_indices, np.int32))
        linesets.append(lineset)
    return linesets


def calculate_trajectories(scene_data, is_fg):
    in_pts = [data['means3D'][is_fg][::traj_frac].contiguous().float().cpu().numpy() for data in scene_data]
    num_lines = len(in_pts[0])
    cols = np.repeat(colormap[np.arange(len(in_pts[0])) % len(colormap)][None], traj_length, 0).reshape(-1, 3)
    out_pts = []
    for t in range(len(in_pts))[traj_length:]:
        out_pts.append(np.array(in_pts[t - traj_length:t + 1]).reshape(-1, 3))
    return make_lineset(out_pts, cols, num_lines)


def calculate_rot_vec(scene_data, is_fg):
    in_pts = [data['means3D'][is_fg][::traj_frac].contiguous().float().cpu().numpy() for data in scene_data]
    in_rotation = [data['rotations'][is_fg][::traj_frac] for data in scene_data]
    num_lines = len(in_pts[0])
    cols = colormap[np.arange(num_lines) % len(colormap)]
    inv_init_q = deepcopy(in_rotation[0])
    inv_init_q[:, 1:] = -1 * inv_init_q[:, 1:]
    inv_init_q = inv_init_q / (inv_init_q ** 2).sum(-1)[:, None]
    init_vec = np.array([-0.1, 0, 0])
    out_pts = []
    for t in range(len(in_pts)):
        cam_rel_qs = quat_mult(in_rotation[t], inv_init_q)
        rot = build_rotation(cam_rel_qs).cpu().numpy()
        vec = (rot @ init_vec[None, :, None]).squeeze()
        out_pts.append(np.concatenate((in_pts[t] + vec, in_pts[t]), 0))
    return make_lineset(out_pts, cols, num_lines)


def render(w2c, k, timestep_data):
    with torch.no_grad():
        # Make a copy of the timestep data that we can modify
        rendervar = timestep_data.copy()
        
        # Apply background removal if enabled
        if ui_state['remove_background']:
            # Get the foreground mask
            is_fg = rendervar['colors_precomp'][:, 0] > 0.5
            # Set opacity to 0 for background points
            mask_opacities = rendervar['opacities'].clone()
            mask_opacities[~is_fg] = 0.0
            rendervar['opacities'] = mask_opacities
            
        cam = setup_camera(w, h, k, w2c, near, far)
        im, _, depth, = Renderer(raster_settings=cam)(**rendervar)
        return im, depth


def rgbd2pcd(im, depth, w2c, k, show_depth=False, project_to_cam_w_scale=None):
    d_near = 1.5
    d_far = 6
    invk = torch.inverse(torch.tensor(k).cuda().float())
    c2w = torch.inverse(torch.tensor(w2c).cuda().float())
    radial_depth = depth[0].reshape(-1)
    def_rays = (invk @ def_pix.T).T
    def_radial_rays = def_rays / torch.linalg.norm(def_rays, ord=2, dim=-1)[:, None]
    pts_cam = def_radial_rays * radial_depth[:, None]
    z_depth = pts_cam[:, 2]
    if project_to_cam_w_scale is not None:
        pts_cam = project_to_cam_w_scale * pts_cam / z_depth[:, None]
    pts4 = torch.concat((pts_cam, pix_ones), 1)
    pts = (c2w @ pts4.T).T[:, :3]
    if show_depth:
        cols = ((z_depth - d_near) / (d_far - d_near))[:, None].repeat(1, 3)
    else:
        cols = torch.permute(im, (1, 2, 0)).reshape(-1, 3)
    pts = o3d.utility.Vector3dVector(pts.contiguous().double().cpu().numpy())
    cols = o3d.utility.Vector3dVector(cols.contiguous().double().cpu().numpy())
    return pts, cols


def visualize(seq, exp):
    scene_data, is_fg = load_scene_data(seq, exp)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=int(w * view_scale), height=int(h * view_scale), visible=True)

    # Register keyboard callbacks
    vis.register_key_callback(ord(" "), toggle_play_callback)
    vis.register_key_callback(ord("."), next_frame_callback)
    vis.register_key_callback(ord(","), prev_frame_callback)
    vis.register_key_callback(ord("B"), toggle_background_callback)
    vis.register_key_callback(ord("C"), lambda vis: change_render_mode_callback(vis, 'color'))
    vis.register_key_callback(ord("D"), lambda vis: change_render_mode_callback(vis, 'depth'))
    vis.register_key_callback(ord("E"), lambda vis: change_render_mode_callback(vis, 'centers'))
    vis.register_key_callback(ord("T"), lambda vis: toggle_lines_callback(vis, 'trajectories'))
    vis.register_key_callback(ord("R"), lambda vis: toggle_lines_callback(vis, 'rotations'))

    # Create help text that will be rendered in the viewer
    print("==== Keyboard Controls ====")
    print("Space: Play/Pause")
    print(".    : Next frame (when paused)")
    print(",    : Previous frame (when paused)")
    print("B    : Toggle background visibility")
    print("C    : Color render mode")
    print("D    : Depth render mode")
    print("E    : Centers render mode")
    print("T    : Toggle trajectory visualization")
    print("R    : Toggle rotation visualization")
    print("=========================")

    w2c, k = init_camera()
    im, depth = render(w2c, k, scene_data[0])
    init_pts, init_cols = rgbd2pcd(im, depth, w2c, k, show_depth=(ui_state['render_mode'] == 'depth'))
    pcd = o3d.geometry.PointCloud()
    pcd.points = init_pts
    pcd.colors = init_cols
    vis.add_geometry(pcd)

    linesets = None
    lines = None
    if ui_state['additional_lines'] is not None:
        if ui_state['additional_lines'] == 'trajectories':
            linesets = calculate_trajectories(scene_data, is_fg)
        elif ui_state['additional_lines'] == 'rotations':
            linesets = calculate_rot_vec(scene_data, is_fg)
        lines = o3d.geometry.LineSet()
        lines.points = linesets[0].points
        lines.colors = linesets[0].colors
        lines.lines = linesets[0].lines
        vis.add_geometry(lines)

    view_k = k * view_scale
    view_k[2, 2] = 1
    view_control = vis.get_view_control()
    cparams = o3d.camera.PinholeCameraParameters()
    cparams.extrinsic = w2c
    cparams.intrinsic.intrinsic_matrix = view_k
    cparams.intrinsic.height = int(h * view_scale)
    cparams.intrinsic.width = int(w * view_scale)
    view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)

    render_options = vis.get_render_option()
    render_options.point_size = view_scale
    render_options.light_on = False
    
    # Create a status text for on-screen information
    status_text = None
    try:
        # Try using Text3D if available in this Open3D version
        status_text = o3d.visualization.rendering.Text3D()
        status_text.text = f"Frame: 0/{len(scene_data)-1} | Mode: {ui_state['render_mode']} | Background: {'Hidden' if ui_state['remove_background'] else 'Visible'}"
        status_text.font_size = 16
        vis.add_geometry(status_text)
    except (AttributeError, ImportError):
        # Text3D isn't available, so print status to console instead
        print(f"Starting visualization. Use keyboard controls to navigate.")

    start_time = time.time()
    ui_state['last_time'] = start_time
    num_timesteps = len(scene_data)
    while True:
        current_time = time.time()
        
        if ui_state['is_playing']:
            passed_time = current_time - start_time
            passed_frames = passed_time * fps
            if ui_state['additional_lines'] == 'trajectories':
                t = int(passed_frames % (num_timesteps - traj_length)) + traj_length  # Skip t that don't have full traj.
            else:
                t = int(passed_frames % num_timesteps)
            ui_state['current_frame'] = t
        else:
            t = ui_state['current_frame'] % num_timesteps
        
        # Update status information
        status_message = f"Frame: {t}/{num_timesteps-1} | Mode: {ui_state['render_mode']} | Background: {'Hidden' if ui_state['remove_background'] else 'Visible'}"
        
        # Update text if available
        if status_text is not None:
            try:
                status_text.text = status_message
                vis.update_geometry(status_text)
            except:
                pass  # Silently fail if text update doesn't work
        
        # Only redraw if needed (when frame changes or settings change)
        need_redraw = (current_time - ui_state['last_time'] > 1.0/fps) or ui_state['need_redraw']
        
        if need_redraw:
            ui_state['last_time'] = current_time
            ui_state['need_redraw'] = False
            
            if FORCE_LOOP:
                num_loops = 1.4
                y_angle = 360*t*num_loops / num_timesteps
                w2c, k = init_camera(y_angle)
                cam_params = view_control.convert_to_pinhole_camera_parameters()
                cam_params.extrinsic = w2c
                view_control.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
            else:  # Interactive control
                cam_params = view_control.convert_to_pinhole_camera_parameters()
                view_k = cam_params.intrinsic.intrinsic_matrix
                k = view_k / view_scale
                k[2, 2] = 1
                w2c = cam_params.extrinsic

            if ui_state['render_mode'] == 'centers':
                pts = o3d.utility.Vector3dVector(scene_data[t]['means3D'].contiguous().double().cpu().numpy())
                cols = o3d.utility.Vector3dVector(scene_data[t]['colors_precomp'].contiguous().double().cpu().numpy())
            else:
                im, depth = render(w2c, k, scene_data[t])
                pts, cols = rgbd2pcd(im, depth, w2c, k, show_depth=(ui_state['render_mode'] == 'depth'))
            pcd.points = pts
            pcd.colors = cols
            vis.update_geometry(pcd)

            if ui_state['additional_lines'] is not None:
                if ui_state['additional_lines'] == 'trajectories':
                    lt = t - traj_length
                else:
                    lt = t
                lines.points = linesets[lt].points
                lines.colors = linesets[lt].colors
                lines.lines = linesets[lt].lines
                vis.update_geometry(lines)

            # Print periodic updates to console for status
            if t % 30 == 0:
                print(status_message)

        if not vis.poll_events():
            break
        vis.update_renderer()

    vis.destroy_window()
    del view_control
    del vis
    del render_options


if __name__ == "__main__":
    exp_name = "exp1"
    # for sequence in ["basketball", "boxes", "football", "juggle", "softball", "tennis"]:
    for sequence in ["basketball"]:
        visualize(sequence, exp_name)
