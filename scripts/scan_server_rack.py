"""
Two-stage LED search for a wall-like dummy rag using the FFW_BG2 wrist camera.

The search strategy is:
1. Use the head camera to detect the red LED and estimate its coarse 3D location.
2. Move the right wrist camera near that location while keeping the camera parallel to the wall.
3. If needed, scan a small local raster around the coarse estimate.
4. Once the wrist camera sees the LED, iteratively center the LED blob in the image.

If the head camera cannot see the LED, the script falls back to a global wrist-camera raster scan.

Usage:
    python scripts/scan_server_rack.py --task RobotisLab-Base-FFW-BG2-IK-Rel-v0 --enable_cameras
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
from typing import NamedTuple

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Two-stage LED search with head-camera coarse search and wrist-camera fine centering.")
parser.add_argument("--task", type=str, default="RobotisLab-Base-FFW-BG2-IK-Rel-v0", help="Task name.")
parser.add_argument("--rack_width", type=float, default=1.5, help="Wall width in meters.")
parser.add_argument("--rack_height", type=float, default=2.2, help="Wall height in meters.")
parser.add_argument("--rack_center_x", type=float, default=0.6, help="Wall plane X position relative to robot.")
parser.add_argument("--rack_center_y", type=float, default=0.0, help="Wall center Y relative to robot.")
parser.add_argument("--rack_bottom_z", type=float, default=0.3, help="Wall bottom Z relative to robot.")
parser.add_argument("--scan_distance", type=float, default=0.45, help="Desired wrist camera standoff from the wall (m).")
parser.add_argument("--fov_coverage", type=float, default=0.7, help="Fraction of wrist FOV used as raster step size.")
parser.add_argument("--coarse_window_scale", type=float, default=1.5, help="Local search window size in wrist-FOV units around the head-camera estimate.")
parser.add_argument("--waypoint_threshold", type=float, default=0.03, help="Position tolerance for waypoint arrival (m).")
parser.add_argument("--orientation_threshold", type=float, default=0.10, help="Orientation tolerance for waypoint arrival (rad).")
parser.add_argument("--hold_steps", type=int, default=15, help="Controller hold steps after each wrist move.")
parser.add_argument("--head_settle_steps", type=int, default=20, help="Steps to settle the robot before the head-camera coarse search.")
parser.add_argument("--red_min", type=int, default=180, help="Minimum R channel value for red detection.")
parser.add_argument("--green_max", type=int, default=80, help="Maximum G channel value for red detection.")
parser.add_argument("--blue_max", type=int, default=80, help="Maximum B channel value for red detection.")
parser.add_argument("--min_red_pixels", type=int, default=50, help="Minimum red pixels for wrist-camera detection.")
parser.add_argument("--head_min_red_pixels", type=int, default=8, help="Minimum red pixels for head-camera coarse detection.")
parser.add_argument("--center_pixel_tolerance", type=float, default=12.0, help="Pixel tolerance for LED centering success.")
parser.add_argument("--max_steps_per_goal", type=int, default=500, help="Maximum control steps allowed for each pose target.")
parser.add_argument("--max_centering_iters", type=int, default=8, help="Maximum wrist-camera centering iterations.")
parser.add_argument("--position_gain", type=float, default=5.0, help="Proportional gain for translational IK action.")
parser.add_argument("--rotation_gain", type=float, default=2.5, help="Proportional gain for rotational IK action.")
parser.add_argument("--rotation_action_limit", type=float, default=0.5, help="Clamp for rotational action components.")
parser.add_argument("--rotation_scale", type=float, default=0.35, help="Scale enabled for rotational IK action components.")
parser.add_argument("--use_head_depth", action="store_true", help="Use head-camera distance_to_image_plane for coarse 3D localization when available.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab.utils.math as math_utils
from isaaclab.sensors import Camera

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import robotis_lab  # noqa: F401


class RedBlob(NamedTuple):
    pixel_count: int
    centroid_u: float
    centroid_v: float
    error_u: float
    error_v: float


def wall_bounds() -> tuple[float, float, float, float]:
    """Return wall bounds as (y_min, y_max, z_min, z_max)."""
    y_min = args_cli.rack_center_y - args_cli.rack_width / 2.0
    y_max = args_cli.rack_center_y + args_cli.rack_width / 2.0
    z_min = args_cli.rack_bottom_z
    z_max = args_cli.rack_bottom_z + args_cli.rack_height
    return y_min, y_max, z_min, z_max


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def compute_camera_view_size(camera: Camera, distance: float) -> tuple[float, float]:
    """Compute the metric width and height visible at a given distance."""
    intrinsics = camera.data.intrinsic_matrices[0]
    image_h, image_w = camera.data.image_shape
    view_width = distance * image_w / float(intrinsics[0, 0].item())
    view_height = distance * image_h / float(intrinsics[1, 1].item())
    return view_width, view_height


def generate_scan_axis(max_val: float, min_val: float, step: float) -> list[float]:
    """Generate evenly spaced scan centers from high to low."""
    if max_val <= min_val:
        return [(max_val + min_val) * 0.5]

    span = max_val - min_val
    if step <= 0.0 or span <= step:
        return [(max_val + min_val) * 0.5]

    count = max(1, math.ceil(span / step))
    actual_step = span / count
    return [max_val - actual_step * (idx + 0.5) for idx in range(count)]


def generate_serpentine_waypoints(
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    x_pos: float,
    y_step: float,
    z_step: float,
) -> list[torch.Tensor]:
    """Generate serpentine camera positions over the wall plane."""
    rows_z = generate_scan_axis(z_max, z_min, z_step)
    cols_y = generate_scan_axis(y_max, y_min, y_step)

    waypoints: list[torch.Tensor] = []
    for row_idx, z_val in enumerate(rows_z):
        scan_cols = cols_y if row_idx % 2 == 0 else list(reversed(cols_y))
        for y_val in scan_cols:
            waypoints.append(torch.tensor([x_pos, y_val, z_val], dtype=torch.float32))
    return waypoints


def make_local_waypoints(center_y: float, center_z: float, wrist_camera: Camera) -> list[torch.Tensor]:
    """Create a small local raster around the head-camera estimate."""
    y_min, y_max, z_min, z_max = wall_bounds()
    wrist_view_width, wrist_view_height = compute_camera_view_size(wrist_camera, args_cli.scan_distance)
    local_width = wrist_view_width * args_cli.coarse_window_scale
    local_height = wrist_view_height * args_cli.coarse_window_scale

    local_y_min = clamp(center_y - local_width * 0.5, y_min, y_max)
    local_y_max = clamp(center_y + local_width * 0.5, y_min, y_max)
    local_z_min = clamp(center_z - local_height * 0.5, z_min, z_max)
    local_z_max = clamp(center_z + local_height * 0.5, z_min, z_max)

    y_step = wrist_view_width * args_cli.fov_coverage
    z_step = wrist_view_height * args_cli.fov_coverage
    return generate_serpentine_waypoints(
        y_min=local_y_min,
        y_max=local_y_max,
        z_min=local_z_min,
        z_max=local_z_max,
        x_pos=args_cli.rack_center_x - args_cli.scan_distance,
        y_step=y_step,
        z_step=z_step,
    )


def make_global_waypoints(wrist_camera: Camera) -> list[torch.Tensor]:
    """Create a global raster across the full wall."""
    y_min, y_max, z_min, z_max = wall_bounds()
    wrist_view_width, wrist_view_height = compute_camera_view_size(wrist_camera, args_cli.scan_distance)
    y_step = wrist_view_width * args_cli.fov_coverage
    z_step = wrist_view_height * args_cli.fov_coverage
    return generate_serpentine_waypoints(
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        z_max=z_max,
        x_pos=args_cli.rack_center_x - args_cli.scan_distance,
        y_step=y_step,
        z_step=z_step,
    )


def get_red_blob(
    camera: Camera,
    red_min: int,
    green_max: int,
    blue_max: int,
    min_red_pixels: int,
) -> tuple[RedBlob | None, torch.Tensor]:
    """Detect a red blob and return its centroid."""
    rgb = camera.data.output["rgb"][0]
    image_h, image_w, _ = rgb.shape

    mask = (
        (rgb[:, :, 0] >= red_min)
        & (rgb[:, :, 1] <= green_max)
        & (rgb[:, :, 2] <= blue_max)
    )

    pixel_count = int(mask.sum().item())
    if pixel_count < min_red_pixels:
        return None, mask

    ys, xs = torch.nonzero(mask, as_tuple=True)
    centroid_u = float(xs.float().mean().item())
    centroid_v = float(ys.float().mean().item())
    center_u = (image_w - 1) * 0.5
    center_v = (image_h - 1) * 0.5

    blob = RedBlob(
        pixel_count=pixel_count,
        centroid_u=centroid_u,
        centroid_v=centroid_v,
        error_u=centroid_u - center_u,
        error_v=centroid_v - center_v,
    )
    return blob, mask


def is_blob_centered(blob: RedBlob) -> bool:
    """Check if the LED blob is centered enough in the image."""
    return abs(blob.error_u) <= args_cli.center_pixel_tolerance and abs(blob.error_v) <= args_cli.center_pixel_tolerance


def pixel_to_ros_ray(camera: Camera, pixel_u: float, pixel_v: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert an image pixel into a world-space ray using ROS optical coordinates."""
    intrinsics = camera.data.intrinsic_matrices[0]
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    device = intrinsics.device

    ray_cam = torch.tensor(
        [
            (pixel_u - float(cx.item())) / float(fx.item()),
            (pixel_v - float(cy.item())) / float(fy.item()),
            1.0,
        ],
        dtype=torch.float32,
        device=device,
    )
    ray_cam = ray_cam / torch.norm(ray_cam)

    origin_w = camera.data.pos_w[0]
    quat_w_ros = camera.data.quat_w_ros[0].unsqueeze(0)
    ray_w = math_utils.quat_apply(quat_w_ros, ray_cam.unsqueeze(0))[0]
    ray_w = ray_w / torch.norm(ray_w)
    return origin_w, ray_w


def reconstruct_world_point_from_depth(camera: Camera, pixel_u: float, pixel_v: float, depth_z: float) -> torch.Tensor:
    """Back-project a pixel with depth into the world frame."""
    intrinsics = camera.data.intrinsic_matrices[0]
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    device = intrinsics.device

    point_cam = torch.tensor(
        [
            (pixel_u - float(cx.item())) / float(fx.item()) * depth_z,
            (pixel_v - float(cy.item())) / float(fy.item()) * depth_z,
            depth_z,
        ],
        dtype=torch.float32,
        device=device,
    )
    point_w = camera.data.pos_w[0] + math_utils.quat_apply(camera.data.quat_w_ros[0].unsqueeze(0), point_cam.unsqueeze(0))[0]
    return point_w


def get_depth_image(camera: Camera) -> torch.Tensor | None:
    """Return the distance_to_image_plane image if available."""
    if "distance_to_image_plane" not in camera.data.output:
        return None
    depth = camera.data.output["distance_to_image_plane"][0]
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return depth


def estimate_led_world_point(
    camera: Camera,
    blob: RedBlob,
    mask: torch.Tensor,
    use_depth: bool,
) -> torch.Tensor | None:
    """Estimate the LED position in world coordinates."""
    if use_depth:
        depth_image = get_depth_image(camera)
        if depth_image is not None:
            depth_values = depth_image[mask]
            depth_values = depth_values[torch.isfinite(depth_values) & (depth_values > 0.0)]
            if depth_values.numel() > 0:
                depth_z = float(depth_values.median().item())
                return reconstruct_world_point_from_depth(camera, blob.centroid_u, blob.centroid_v, depth_z)

    ray_origin_w, ray_dir_w = pixel_to_ros_ray(camera, blob.centroid_u, blob.centroid_v)
    dx = float(ray_dir_w[0].item())
    if abs(dx) < 1e-6:
        return None

    ray_scale = (args_cli.rack_center_x - float(ray_origin_w[0].item())) / dx
    if ray_scale <= 0.0:
        return None

    point_w = ray_origin_w + ray_dir_w * ray_scale
    y_min, y_max, z_min, z_max = wall_bounds()
    point_w[1] = clamp(float(point_w[1].item()), y_min, y_max)
    point_w[2] = clamp(float(point_w[2].item()), z_min, z_max)
    return point_w


def plan_camera_position_from_wall_point(point_w: torch.Tensor) -> torch.Tensor:
    """Convert a wall point into the desired wrist-camera position."""
    y_min, y_max, z_min, z_max = wall_bounds()
    return torch.tensor(
        [
            args_cli.rack_center_x - args_cli.scan_distance,
            clamp(float(point_w[1].item()), y_min, y_max),
            clamp(float(point_w[2].item()), z_min, z_max),
        ],
        dtype=torch.float32,
        device=point_w.device,
    )


def build_fixed_wall_camera_quat(device: torch.device | str) -> torch.Tensor:
    """Return a constant camera orientation that faces the wall with +X forward and +Z up."""
    forward = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device)
    up_hint = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
    z_axis = up_hint - torch.dot(up_hint, forward) * forward
    z_axis = z_axis / torch.norm(z_axis)
    y_axis = torch.cross(z_axis, forward, dim=0)
    y_axis = y_axis / torch.norm(y_axis)
    rot_mat = torch.stack([forward, y_axis, z_axis], dim=1).unsqueeze(0)
    return math_utils.quat_from_matrix(rot_mat)[0]


def compute_camera_mount_in_ee(
    ee_pos_w: torch.Tensor,
    ee_quat_w: torch.Tensor,
    cam_pos_w: torch.Tensor,
    cam_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the rigid camera transform in the end-effector frame."""
    ee_rot_w = math_utils.matrix_from_quat(ee_quat_w.unsqueeze(0))[0]
    cam_rot_w = math_utils.matrix_from_quat(cam_quat_w.unsqueeze(0))[0]
    rot_ee_cam = ee_rot_w.transpose(0, 1) @ cam_rot_w
    pos_ee_cam = ee_rot_w.transpose(0, 1) @ (cam_pos_w - ee_pos_w)
    return pos_ee_cam, rot_ee_cam


def camera_pose_to_ee_pose(
    target_cam_pos_w: torch.Tensor,
    target_cam_quat_w: torch.Tensor,
    cam_pos_ee: torch.Tensor,
    rot_ee_cam: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a desired camera pose into the corresponding end-effector pose."""
    cam_rot_w = math_utils.matrix_from_quat(target_cam_quat_w.unsqueeze(0))[0]
    ee_rot_w = cam_rot_w @ rot_ee_cam.transpose(0, 1)
    ee_pos_w = target_cam_pos_w - ee_rot_w @ cam_pos_ee
    ee_quat_w = math_utils.quat_from_matrix(ee_rot_w.unsqueeze(0))[0]
    return ee_pos_w, ee_quat_w


def compute_pose_action(
    current_pos_w: torch.Tensor,
    current_quat_w: torch.Tensor,
    target_pos_w: torch.Tensor,
    target_quat_w: torch.Tensor,
    action_dim: int,
) -> torch.Tensor:
    """Compute a relative pose action for the IK controller."""
    position_error = target_pos_w - current_pos_w
    position_action = torch.clamp(position_error * args_cli.position_gain, -1.0, 1.0)

    current_rot_w = math_utils.matrix_from_quat(current_quat_w.unsqueeze(0))[0]
    target_rot_w = math_utils.matrix_from_quat(target_quat_w.unsqueeze(0))[0]
    delta_rot = target_rot_w @ current_rot_w.transpose(0, 1)
    delta_quat = math_utils.quat_from_matrix(delta_rot.unsqueeze(0))
    rotation_error = math_utils.axis_angle_from_quat(delta_quat)[0]
    rotation_action = torch.clamp(
        rotation_error * args_cli.rotation_gain,
        -args_cli.rotation_action_limit,
        args_cli.rotation_action_limit,
    )

    if action_dim < 7:
        raise ValueError(f"Expected action_dim >= 7 for arm IK + gripper control, but received {action_dim}.")

    action = torch.zeros(action_dim, dtype=torch.float32, device=current_pos_w.device)
    action[0:3] = position_action
    action[3:6] = rotation_action
    action[6] = 1.0
    return action


def step_pose_hold(
    env,
    ee_frame,
    target_pos_w: torch.Tensor,
    target_quat_w: torch.Tensor,
    steps: int,
    action_dim: int,
) -> int:
    """Hold a desired pose for a few controller steps."""
    executed_steps = 0
    for _ in range(steps):
        if not simulation_app.is_running():
            break
        current_pos_w = ee_frame.data.target_pos_w[0, 0, :]
        current_quat_w = ee_frame.data.target_quat_w[0, 0, :]
        action = compute_pose_action(current_pos_w, current_quat_w, target_pos_w, target_quat_w, action_dim)
        env.step(action.unsqueeze(0))
        executed_steps += 1
    return executed_steps


def move_ee_to_camera_pose(
    env,
    ee_frame,
    target_cam_pos_w: torch.Tensor,
    fixed_cam_quat_w: torch.Tensor,
    cam_pos_ee: torch.Tensor,
    rot_ee_cam: torch.Tensor,
    action_dim: int,
) -> tuple[bool, int, float, float, torch.Tensor, torch.Tensor]:
    """Move the end-effector so the wrist camera reaches the desired pose."""
    target_ee_pos_w, target_ee_quat_w = camera_pose_to_ee_pose(target_cam_pos_w, fixed_cam_quat_w, cam_pos_ee, rot_ee_cam)
    steps = 0
    pos_error = float("inf")
    rot_error = float("inf")

    while simulation_app.is_running() and steps < args_cli.max_steps_per_goal:
        current_pos_w = ee_frame.data.target_pos_w[0, 0, :]
        current_quat_w = ee_frame.data.target_quat_w[0, 0, :]
        pos_error = float(torch.norm(target_ee_pos_w - current_pos_w).item())
        rot_error = float(math_utils.quat_error_magnitude(current_quat_w.unsqueeze(0), target_ee_quat_w.unsqueeze(0))[0].item())

        if pos_error <= args_cli.waypoint_threshold and rot_error <= args_cli.orientation_threshold:
            return True, steps, pos_error, rot_error, target_ee_pos_w, target_ee_quat_w

        action = compute_pose_action(current_pos_w, current_quat_w, target_ee_pos_w, target_ee_quat_w, action_dim)
        env.step(action.unsqueeze(0))
        steps += 1

    return False, steps, pos_error, rot_error, target_ee_pos_w, target_ee_quat_w


def scan_waypoints(
    env,
    ee_frame,
    wrist_camera: Camera,
    waypoints: list[torch.Tensor],
    fixed_cam_quat_w: torch.Tensor,
    cam_pos_ee: torch.Tensor,
    rot_ee_cam: torch.Tensor,
    action_dim: int,
    label: str,
) -> tuple[bool, torch.Tensor | None, RedBlob | None, torch.Tensor | None, int]:
    """Move through camera waypoints until the wrist camera detects the LED."""
    steps_used = 0
    for idx, camera_target in enumerate(waypoints):
        camera_target = camera_target.to(device=fixed_cam_quat_w.device)
        print(
            f"[{label} {idx + 1:02d}/{len(waypoints):02d}] "
            f"cam=({camera_target[0].item():.3f}, {camera_target[1].item():.3f}, {camera_target[2].item():.3f})"
        )
        reached, move_steps, pos_error, rot_error, target_ee_pos_w, target_ee_quat_w = move_ee_to_camera_pose(
            env,
            ee_frame,
            camera_target,
            fixed_cam_quat_w,
            cam_pos_ee,
            rot_ee_cam,
            action_dim,
        )
        steps_used += move_steps

        hold_steps = step_pose_hold(env, ee_frame, target_ee_pos_w, target_ee_quat_w, args_cli.hold_steps, action_dim)
        steps_used += hold_steps

        if not reached:
            print(f"  pose not fully reached: pos_err={pos_error:.4f}m rot_err={rot_error:.4f}rad")

        blob, mask = get_red_blob(
            wrist_camera,
            red_min=args_cli.red_min,
            green_max=args_cli.green_max,
            blue_max=args_cli.blue_max,
            min_red_pixels=args_cli.min_red_pixels,
        )
        if blob is None:
            print("  wrist camera: no LED")
            continue

        print(
            f"  wrist camera: detected {blob.pixel_count} red pixels "
            f"at ({blob.centroid_u:.1f}, {blob.centroid_v:.1f})"
        )
        return True, camera_target, blob, mask, steps_used

    return False, None, None, None, steps_used


def fine_center_led(
    env,
    ee_frame,
    wrist_camera: Camera,
    start_camera_pos_w: torch.Tensor,
    fixed_cam_quat_w: torch.Tensor,
    cam_pos_ee: torch.Tensor,
    rot_ee_cam: torch.Tensor,
    action_dim: int,
) -> tuple[bool, torch.Tensor, int]:
    """Use the wrist camera to iteratively center the LED."""
    steps_used = 0
    target_camera_pos_w = start_camera_pos_w.clone()

    for iter_idx in range(args_cli.max_centering_iters):
        print(f"[fine {iter_idx + 1:02d}/{args_cli.max_centering_iters:02d}] centering with wrist camera")
        reached, move_steps, pos_error, rot_error, target_ee_pos_w, target_ee_quat_w = move_ee_to_camera_pose(
            env,
            ee_frame,
            target_camera_pos_w,
            fixed_cam_quat_w,
            cam_pos_ee,
            rot_ee_cam,
            action_dim,
        )
        steps_used += move_steps

        hold_steps = step_pose_hold(env, ee_frame, target_ee_pos_w, target_ee_quat_w, args_cli.hold_steps, action_dim)
        steps_used += hold_steps

        blob, mask = get_red_blob(
            wrist_camera,
            red_min=args_cli.red_min,
            green_max=args_cli.green_max,
            blue_max=args_cli.blue_max,
            min_red_pixels=args_cli.min_red_pixels,
        )
        if blob is None:
            print("  lost LED in wrist camera during centering")
            return False, target_camera_pos_w, steps_used

        print(
            f"  centroid=({blob.centroid_u:.1f}, {blob.centroid_v:.1f}) "
            f"error=({blob.error_u:.1f}, {blob.error_v:.1f})"
        )

        if reached and is_blob_centered(blob):
            print("  LED centered successfully")
            return True, target_camera_pos_w, steps_used

        led_point_w = estimate_led_world_point(wrist_camera, blob, mask, use_depth=False)
        if led_point_w is None:
            print("  could not back-project wrist observation onto wall")
            return False, target_camera_pos_w, steps_used

        target_camera_pos_w = plan_camera_position_from_wall_point(led_point_w)

        if not reached:
            print(f"  continuing despite pose error: pos_err={pos_error:.4f}m rot_err={rot_error:.4f}rad")

    return False, target_camera_pos_w, steps_used


def main():
    print("=" * 60)
    print("Two-Stage Dummy Rag LED Search")
    print("=" * 60)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.terminations.time_out = None
    env_cfg.terminations.success = None
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.actions.arm_action.scale = (
        0.3,
        0.3,
        0.3,
        args_cli.rotation_scale,
        args_cli.rotation_scale,
        args_cli.rotation_scale,
    )
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    action_dim = env.action_manager.total_action_dim

    env.sim.reset()
    env.reset()

    wrist_camera: Camera = env.scene["right_wrist_cam"]
    head_camera: Camera = env.scene["head_cam"]
    ee_frame = env.scene["ee_frame"]

    total_steps = 0
    coarse_point_w = None
    coarse_mode = "none"

    initial_ee_pos_w = ee_frame.data.target_pos_w[0, 0, :]
    initial_ee_quat_w = ee_frame.data.target_quat_w[0, 0, :]
    total_steps += step_pose_hold(env, ee_frame, initial_ee_pos_w, initial_ee_quat_w, args_cli.head_settle_steps, action_dim)

    fixed_cam_quat_w = build_fixed_wall_camera_quat(env.device)
    cam_pos_ee, rot_ee_cam = compute_camera_mount_in_ee(
        ee_pos_w=ee_frame.data.target_pos_w[0, 0, :],
        ee_quat_w=ee_frame.data.target_quat_w[0, 0, :],
        cam_pos_w=wrist_camera.data.pos_w[0],
        cam_quat_w=wrist_camera.data.quat_w_world[0],
    )

    wrist_view_width, wrist_view_height = compute_camera_view_size(wrist_camera, args_cli.scan_distance)
    print(f"Wrist camera view at {args_cli.scan_distance:.3f}m: {wrist_view_width:.3f}m x {wrist_view_height:.3f}m")

    with torch.inference_mode():
        head_blob, head_mask = get_red_blob(
            head_camera,
            red_min=args_cli.red_min,
            green_max=args_cli.green_max,
            blue_max=args_cli.blue_max,
            min_red_pixels=args_cli.head_min_red_pixels,
        )

        if head_blob is not None:
            coarse_point_w = estimate_led_world_point(
                head_camera,
                head_blob,
                head_mask,
                use_depth=args_cli.use_head_depth,
            )
            if coarse_point_w is not None:
                coarse_mode = "head_depth" if args_cli.use_head_depth else "head_plane"
                print(
                    f"Head camera coarse localization ({coarse_mode}): "
                    f"point=({coarse_point_w[0].item():.3f}, {coarse_point_w[1].item():.3f}, {coarse_point_w[2].item():.3f})"
                )
            else:
                print("Head camera saw red but could not estimate a coarse 3D point.")
        else:
            print("Head camera could not find the LED. Falling back to wrist-camera raster search.")

        found = False
        found_camera_pos_w = None
        search_blob = None
        search_mask = None

        if coarse_point_w is not None:
            coarse_camera_pos_w = plan_camera_position_from_wall_point(coarse_point_w)
            print(
                "Moving wrist camera to coarse target: "
                f"({coarse_camera_pos_w[0].item():.3f}, {coarse_camera_pos_w[1].item():.3f}, {coarse_camera_pos_w[2].item():.3f})"
            )
            reached, move_steps, pos_error, rot_error, target_ee_pos_w, target_ee_quat_w = move_ee_to_camera_pose(
                env,
                ee_frame,
                coarse_camera_pos_w,
                fixed_cam_quat_w,
                cam_pos_ee,
                rot_ee_cam,
                action_dim,
            )
            total_steps += move_steps
            total_steps += step_pose_hold(
                env, ee_frame, target_ee_pos_w, target_ee_quat_w, args_cli.hold_steps, action_dim
            )

            if not reached:
                print(f"Coarse move pose error: pos_err={pos_error:.4f}m rot_err={rot_error:.4f}rad")

            search_blob, search_mask = get_red_blob(
                wrist_camera,
                red_min=args_cli.red_min,
                green_max=args_cli.green_max,
                blue_max=args_cli.blue_max,
                min_red_pixels=args_cli.min_red_pixels,
            )
            if search_blob is not None:
                found = True
                found_camera_pos_w = coarse_camera_pos_w
                print("Wrist camera acquired the LED directly from the coarse move.")
            else:
                print("Wrist camera did not acquire the LED directly. Running local raster search.")
                local_waypoints = make_local_waypoints(float(coarse_point_w[1].item()), float(coarse_point_w[2].item()), wrist_camera)
                found, found_camera_pos_w, search_blob, search_mask, used_steps = scan_waypoints(
                    env,
                    ee_frame,
                    wrist_camera,
                    local_waypoints,
                    fixed_cam_quat_w,
                    cam_pos_ee,
                    rot_ee_cam,
                    action_dim,
                    label="local",
                )
                total_steps += used_steps

        if not found:
            print("Running global wrist-camera raster search.")
            global_waypoints = make_global_waypoints(wrist_camera)
            found, found_camera_pos_w, search_blob, search_mask, used_steps = scan_waypoints(
                env,
                ee_frame,
                wrist_camera,
                global_waypoints,
                fixed_cam_quat_w,
                cam_pos_ee,
                rot_ee_cam,
                action_dim,
                label="global",
            )
            total_steps += used_steps

        centered = False
        centered_camera_pos_w = found_camera_pos_w
        if found and found_camera_pos_w is not None and search_blob is not None:
            if is_blob_centered(search_blob):
                centered = True
                print("Wrist camera detection was already centered.")
            else:
                centered, centered_camera_pos_w, used_steps = fine_center_led(
                    env,
                    ee_frame,
                    wrist_camera,
                    found_camera_pos_w,
                    fixed_cam_quat_w,
                    cam_pos_ee,
                    rot_ee_cam,
                    action_dim,
                )
                total_steps += used_steps

        print("\n" + "=" * 60)
        print("SEARCH COMPLETE")
        print("=" * 60)
        print(f"Coarse stage: {coarse_mode}")
        print(f"Total simulation steps: {total_steps}")
        if centered and centered_camera_pos_w is not None:
            print(
                "LED centered with wrist camera at "
                f"cam=({centered_camera_pos_w[0].item():.3f}, {centered_camera_pos_w[1].item():.3f}, {centered_camera_pos_w[2].item():.3f})"
            )
        elif found and found_camera_pos_w is not None:
            print(
                "LED found but not fully centered at "
                f"cam=({found_camera_pos_w[0].item():.3f}, {found_camera_pos_w[1].item():.3f}, {found_camera_pos_w[2].item():.3f})"
            )
        else:
            print("No LED detected during the search.")
        print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
