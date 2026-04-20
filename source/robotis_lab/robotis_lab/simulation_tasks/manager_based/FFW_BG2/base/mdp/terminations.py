from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def red_led_in_center(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("right_wrist_cam"),
    center_ratio: float = 0.2,
    red_min: int = 180,
    green_max: int = 80,
    blue_max: int = 80,
    min_red_pixels: int = 50,
    min_center_coverage: float = 0.8,
) -> torch.Tensor:
    """Check if the red LED itself is centered in the wrist-camera image.

    The LED must satisfy all of the following:
    1. Enough red pixels are detected in the full image.
    2. The red-pixel centroid lies inside a central window.
    3. Most of the detected red pixels lie inside that central window.

    Args:
        env: The RL environment instance.
        sensor_cfg: Configuration for the camera sensor.
        center_ratio: Fraction of the image (width and height) to use as center region.
        red_min: Minimum R channel value to be considered red.
        green_max: Maximum G channel value to be considered red.
        blue_max: Maximum B channel value to be considered red.
        min_red_pixels: Minimum number of red pixels in the full image for success.
        min_center_coverage: Minimum fraction of detected red pixels that must lie inside the center region.

    Returns:
        Boolean tensor indicating which environments have red LED in center.
    """
    camera: Camera = env.scene[sensor_cfg.name]
    rgb = camera.data.output["rgb"][..., :3]  # (N, H, W, 3), uint8/float-like

    _, h, w, _ = rgb.shape

    # Detect red pixels in the full image.
    is_red = (
        (rgb[:, :, :, 0] >= red_min)
        & (rgb[:, :, :, 1] <= green_max)
        & (rgb[:, :, :, 2] <= blue_max)
    )
    total_red_pixel_count = is_red.sum(dim=(1, 2))

    # Compute center crop boundaries.
    margin_h = int(h * (1.0 - center_ratio) / 2.0)
    margin_w = int(w * (1.0 - center_ratio) / 2.0)
    center_red = is_red[:, margin_h : h - margin_h, margin_w : w - margin_w]
    center_red_pixel_count = center_red.sum(dim=(1, 2))

    # Compute the red-pixel centroid over the full image.
    red_weights = is_red.to(dtype=torch.float32)
    y_coords = torch.arange(h, device=rgb.device, dtype=torch.float32).view(1, h, 1)
    x_coords = torch.arange(w, device=rgb.device, dtype=torch.float32).view(1, 1, w)
    total_red_weight = red_weights.sum(dim=(1, 2)).clamp_min(1.0)
    centroid_y = (red_weights * y_coords).sum(dim=(1, 2)) / total_red_weight
    centroid_x = (red_weights * x_coords).sum(dim=(1, 2)) / total_red_weight

    # The LED must be centered, not just partially overlap the region.
    centroid_in_center = (
        (centroid_y >= margin_h)
        & (centroid_y < (h - margin_h))
        & (centroid_x >= margin_w)
        & (centroid_x < (w - margin_w))
    )
    center_coverage = center_red_pixel_count.to(torch.float32) / total_red_weight

    return (
        (total_red_pixel_count >= min_red_pixels)
        & centroid_in_center
        & (center_coverage >= min_center_coverage)
    )
