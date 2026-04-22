"""Record and restore the non-physics LedTargetAnchor pose for replay fidelity."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass


LED_TARGET_ANCHOR_EXPR = "/World/envs/env_.*/DummyRag/LedTargetAnchor"
LED_TARGET_ANCHOR_NAME = "led_target_anchor"
LED_TARGET_ANCHOR_PENDING_ATTR = "_pending_led_target_anchor_root_pose"


def _env_ids_tensor(env, env_ids: Sequence[int] | torch.Tensor | slice | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, dtype=torch.long, device=env.device)[env_ids]
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long)
    return torch.tensor(env_ids, dtype=torch.long, device=env.device)


def _index_env_values(env, value, env_ids: Sequence[int] | torch.Tensor | slice | None):
    if isinstance(value, dict):
        return {key: _index_env_values(env, sub_value, env_ids) for key, sub_value in value.items()}
    ids = _env_ids_tensor(env, env_ids).to(device=value.device)
    return value[ids].clone()


def _ensure_led_target_anchor_view(env) -> bool:
    if getattr(env, "_led_target_anchor_view", None) is not None:
        return True

    ensure_method = getattr(env, "_ensure_led_target_anchor_view", None)
    if ensure_method is not None:
        ensure_method()
        if getattr(env, "_led_target_anchor_view", None) is not None:
            return True

    from isaacsim.core.prims import XFormPrim

    candidate_exprs = [LED_TARGET_ANCHOR_EXPR]
    if "dummy_rag" in env.scene.keys():
        asset_root_expr = env.scene["dummy_rag"].cfg.prim_path.rstrip("/")
        candidate_exprs.append(f"{asset_root_expr}/LedTargetAnchor")

    for expr in candidate_exprs:
        prim_paths = sim_utils.find_matching_prim_paths(expr)
        if prim_paths:
            env._led_target_anchor_view = XFormPrim(expr, reset_xform_properties=False)
            env._led_target_anchor_expr = expr
            env._led_target_anchor_prim_paths = prim_paths
            return True
    return False


def get_led_target_anchor_root_pose(
    env,
    env_ids: Sequence[int] | torch.Tensor | slice | None = None,
) -> torch.Tensor | None:
    """Return LedTargetAnchor root pose as env-relative ``(x, y, z, qw, qx, qy, qz)``."""

    if not _ensure_led_target_anchor_view(env):
        return None

    ids = _env_ids_tensor(env, env_ids)
    anchor_pos_w, anchor_quat_w = env._led_target_anchor_view.get_world_poses()
    pose_ids = ids.to(device=anchor_pos_w.device)
    origin_ids = ids.to(device=env.scene.env_origins.device)
    anchor_pos = anchor_pos_w[pose_ids] - env.scene.env_origins[origin_ids, :3].to(device=anchor_pos_w.device)
    anchor_quat = anchor_quat_w[pose_ids]
    return torch.cat([anchor_pos, anchor_quat], dim=-1).clone()


def get_scene_state_with_led_target_anchor(
    env,
    env_ids: Sequence[int] | torch.Tensor | slice | None = None,
) -> dict:
    """Return Isaac Lab scene state plus the non-physics LedTargetAnchor XForm pose."""

    state = _index_env_values(env, env.scene.get_state(is_relative=True), env_ids)
    root_pose = get_led_target_anchor_root_pose(env, env_ids)
    if root_pose is not None:
        state.setdefault("xform", {})[LED_TARGET_ANCHOR_NAME] = {"root_pose": root_pose}
    return state


def restore_led_target_anchor_root_pose(
    env,
    root_pose: torch.Tensor,
    env_ids: Sequence[int] | torch.Tensor | slice | None = None,
) -> bool:
    """Restore LedTargetAnchor from an env-relative root pose tensor."""

    if root_pose is None or not _ensure_led_target_anchor_view(env):
        return False

    ids = _env_ids_tensor(env, env_ids)
    root_pose = torch.as_tensor(root_pose, dtype=torch.float32, device=env.device)
    if root_pose.ndim == 1:
        root_pose = root_pose.unsqueeze(0)
    if root_pose.shape[0] == 1 and ids.numel() > 1:
        root_pose = root_pose.repeat(ids.numel(), 1)
    if root_pose.shape[0] != ids.numel():
        raise ValueError(
            f"LedTargetAnchor pose count ({root_pose.shape[0]}) does not match env id count ({ids.numel()})."
        )

    origin_ids = ids.to(device=env.scene.env_origins.device)
    pos_w = root_pose[:, :3] + env.scene.env_origins[origin_ids, :3].to(device=root_pose.device)
    quat_w = root_pose[:, 3:7]
    env._led_target_anchor_view.set_world_poses(pos_w, quat_w, ids.detach().cpu().long())
    return True


def _matrix_pose_to_root_pose(pose_matrix: torch.Tensor) -> torch.Tensor:
    pose_matrix = torch.as_tensor(pose_matrix)
    if pose_matrix.ndim == 2:
        pose_matrix = pose_matrix.unsqueeze(0)
    pos, rot = math_utils.unmake_pose(pose_matrix)
    quat = math_utils.quat_from_matrix(rot)
    return torch.cat([pos, quat], dim=-1)


def extract_led_target_anchor_root_pose_from_initial_state(initial_state: dict) -> torch.Tensor | None:
    xform_state = initial_state.get("xform", {})
    anchor_state = xform_state.get(LED_TARGET_ANCHOR_NAME, {})
    return anchor_state.get("root_pose")


def extract_led_target_anchor_root_pose_from_datagen_info(episode_data: dict) -> torch.Tensor | None:
    try:
        pose_matrix = episode_data["obs"]["datagen_info"]["object_pose"][LED_TARGET_ANCHOR_NAME][0]
    except KeyError:
        return None
    return _matrix_pose_to_root_pose(pose_matrix)


def get_episode_led_target_anchor_root_pose(episode_data: dict) -> tuple[torch.Tensor | None, str | None]:
    initial_state = episode_data.get("initial_state", {})
    root_pose = extract_led_target_anchor_root_pose_from_initial_state(initial_state)
    if root_pose is not None:
        return root_pose, "initial_state/xform/led_target_anchor/root_pose"

    root_pose = extract_led_target_anchor_root_pose_from_datagen_info(episode_data)
    if root_pose is not None:
        return root_pose, "obs/datagen_info/object_pose/led_target_anchor[0]"

    return None, None


def queue_led_target_anchor_restore_from_episode(env, episode_data: dict) -> str | None:
    root_pose, source = get_episode_led_target_anchor_root_pose(episode_data)
    if root_pose is None:
        return None
    setattr(env, LED_TARGET_ANCHOR_PENDING_ATTR, root_pose.detach().clone())
    return source


def consume_queued_led_target_anchor_restore(
    env,
    env_ids: Sequence[int] | torch.Tensor | slice | None = None,
) -> bool:
    root_pose = getattr(env, LED_TARGET_ANCHOR_PENDING_ATTR, None)
    if root_pose is None:
        return False
    setattr(env, LED_TARGET_ANCHOR_PENDING_ATTR, None)
    return restore_led_target_anchor_root_pose(env, root_pose, env_ids)


def restore_led_target_anchor_from_episode(
    env,
    episode_data: dict,
    env_ids: Sequence[int] | torch.Tensor | slice | None = None,
) -> str | None:
    root_pose, source = get_episode_led_target_anchor_root_pose(episode_data)
    if root_pose is None:
        return None
    if not restore_led_target_anchor_root_pose(env, root_pose, env_ids):
        return None
    return source


class LedTargetAnchorInitialStateRecorder(RecorderTerm):
    """Record scene initial state, including the non-physics LedTargetAnchor pose."""

    def record_post_reset(self, env_ids: Sequence[int] | None):
        consume_queued_led_target_anchor_restore(self._env, env_ids)
        return "initial_state", get_scene_state_with_led_target_anchor(self._env, env_ids)


@configclass
class LedTargetAnchorInitialStateRecorderCfg(RecorderTermCfg):
    """Configuration for recording initial state with LedTargetAnchor pose."""

    class_type: type[RecorderTerm] = LedTargetAnchorInitialStateRecorder


class LedTargetAnchorPostStepStatesRecorder(RecorderTerm):
    """Record step states, including the non-physics LedTargetAnchor pose."""

    def record_post_step(self):
        return "states", get_scene_state_with_led_target_anchor(self._env)


@configclass
class LedTargetAnchorPostStepStatesRecorderCfg(RecorderTermCfg):
    """Configuration for recording step states with LedTargetAnchor pose."""

    class_type: type[RecorderTerm] = LedTargetAnchorPostStepStatesRecorder
