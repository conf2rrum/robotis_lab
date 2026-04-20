"""FFW_BG2-specific base environment: robot in an empty room with IK control."""

import isaaclab.sim as sim_utils
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from robotis_lab.simulation_tasks.manager_based.FFW_BG2.base import mdp
from robotis_lab.simulation_tasks.manager_based.FFW_BG2.base.base_env_cfg import BaseEnvCfg

from robotis_lab.assets.robots.FFW_BG2 import FFW_BG2_WITHOUT_MIMIC_CFG  # isort: skip
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


def set_default_joint_pose(env, env_ids, joint_positions, asset_cfg=SceneEntityCfg("robot")):
    """Set the default joint positions for the robot using joint names."""
    import torch
    from isaaclab.assets import Articulation

    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_list = [joint_positions.get(name, 0.0) for name in asset.data.joint_names]
    default_pose = torch.tensor(joint_pos_list, dtype=torch.float32, device=env.device)
    if default_pose.dim() == 1:
        default_pose = default_pose.unsqueeze(0).repeat(len(env_ids), 1)

    asset.data.default_joint_pos[env_ids] = default_pose
    asset.set_joint_position_target(default_pose, env_ids=env_ids)
    asset.write_joint_state_to_sim(default_pose, torch.zeros_like(default_pose), env_ids=env_ids)


def randomize_led_target_anchor_pose(
    env,
    env_ids,
    offset_range,
    led_anchor_prim_expr="/World/envs/env_.*/DummyRag/LedTargetAnchor",
    rag_body_prim_expr="/World/envs/env_.*/DummyRag",
    led_anchor_relative_prim_path="LedTargetAnchor",
    rag_body_relative_prim_path="DummyRagBody",
    asset_cfg=SceneEntityCfg("dummy_rag"),
):
    """Randomize the LED anchor pose while keeping it on the rag-body face.

    The anchor is expected to live outside the rigid-body subtree. We keep its authored
    local offset from the rag body as the surface reference and randomize only around that
    default local pose.
    """
    import torch
    from isaacsim.core.prims import XFormPrim

    import isaaclab.utils.math as math_utils

    if env_ids is None:
        return

    env_ids_cpu = env_ids.detach().cpu().long()

    candidate_expr_pairs: list[tuple[str, str]] = []
    if led_anchor_prim_expr and rag_body_prim_expr:
        candidate_expr_pairs.append((led_anchor_prim_expr, rag_body_prim_expr))

    if asset_cfg is not None and asset_cfg.name in env.scene.keys():
        asset = env.scene[asset_cfg.name]
        asset_root_expr = asset.cfg.prim_path.rstrip("/")
        # Prefer the asset root as the reference frame. This matches the common case
        # where LedTargetAnchor is a non-physics sibling under /DummyRag.
        candidate_expr_pairs.append((f"{asset_root_expr}/{led_anchor_relative_prim_path.strip('/')}", asset_root_expr))
        candidate_expr_pairs.append(
            (
                f"{asset_root_expr}/{led_anchor_relative_prim_path.strip('/')}",
                f"{asset_root_expr}/{rag_body_relative_prim_path.strip('/')}",
            )
        )

    chosen_led_expr = None
    chosen_rag_expr = None
    chosen_led_paths = None
    chosen_rag_paths = None
    debug_attempts: list[str] = []
    for cur_led_expr, cur_rag_expr in candidate_expr_pairs:
        cur_led_paths = sim_utils.find_matching_prim_paths(cur_led_expr)
        cur_rag_paths = sim_utils.find_matching_prim_paths(cur_rag_expr)
        debug_attempts.append(
            f"(anchor='{cur_led_expr}' -> {len(cur_led_paths)} match(es), rag='{cur_rag_expr}' -> {len(cur_rag_paths)} match(es))"
        )
        if cur_led_paths and cur_rag_paths:
            chosen_led_expr = cur_led_expr
            chosen_rag_expr = cur_rag_expr
            chosen_led_paths = cur_led_paths
            chosen_rag_paths = cur_rag_paths
            break

    if chosen_led_expr is None or chosen_rag_expr is None:
        print(
            "[LedTargetAnchor] no compatible anchor/rag prim pair found. "
            f"Attempts: {debug_attempts}"
        )
        return

    views_need_init = (
        not hasattr(env, "_led_target_anchor_view")
        or getattr(env, "_led_target_anchor_expr", None) != chosen_led_expr
        or getattr(env, "_rag_body_expr", None) != chosen_rag_expr
    )

    if views_need_init:
        env._led_target_anchor_view = XFormPrim(chosen_led_expr, reset_xform_properties=False)
        env._rag_body_view = XFormPrim(chosen_rag_expr, reset_xform_properties=False)
        env._led_target_anchor_prim_paths = chosen_led_paths
        env._rag_body_prim_paths = chosen_rag_paths
        env._led_target_anchor_expr = chosen_led_expr
        env._rag_body_expr = chosen_rag_expr

        rag_pos_w, rag_quat_w = env._rag_body_view.get_world_poses()
        led_pos_w, led_quat_w = env._led_target_anchor_view.get_world_poses()
        env._led_target_anchor_default_local_pos = math_utils.quat_apply_inverse(
            rag_quat_w, led_pos_w - rag_pos_w
        ).clone()
        env._led_target_anchor_default_local_quat = math_utils.quat_mul(
            math_utils.quat_conjugate(rag_quat_w), led_quat_w
        ).clone()
        print(
            "[LedTargetAnchor] initialized runtime views with "
            f"anchor expr '{chosen_led_expr}' and rag-body expr '{chosen_rag_expr}'"
        )

    range_x = offset_range.get("x", (0.0, 0.0))
    range_y = offset_range.get("y", (0.0, 0.0))
    range_z = offset_range.get("z", (0.0, 0.0))

    rag_pos_w, rag_quat_w = env._rag_body_view.get_world_poses()
    default_local_pos = env._led_target_anchor_default_local_pos[env_ids_cpu]
    default_local_quat = env._led_target_anchor_default_local_quat[env_ids_cpu]
    rag_pos_w = rag_pos_w[env_ids_cpu]
    rag_quat_w = rag_quat_w[env_ids_cpu]

    offsets = torch.empty(
        (len(env_ids_cpu), 3),
        dtype=default_local_pos.dtype,
        device=default_local_pos.device,
    )
    offsets[:, 0].uniform_(*range_x)
    offsets[:, 1].uniform_(*range_y)
    offsets[:, 2].uniform_(*range_z)

    new_local_pos = default_local_pos + offsets
    new_world_pos = rag_pos_w + math_utils.quat_apply(rag_quat_w, new_local_pos)
    new_world_quat = math_utils.quat_mul(rag_quat_w, default_local_quat)

    env._led_target_anchor_view.set_world_poses(new_world_pos, new_world_quat, env_ids_cpu)

    for i, cur_env in enumerate(env_ids_cpu.tolist()):
        prim_path = (
            env._led_target_anchor_prim_paths[cur_env]
            if cur_env < len(env._led_target_anchor_prim_paths)
            else f"<env {cur_env}>"
        )
        print(
            "[LedTargetAnchor] "
            f"env {cur_env}: randomized {prim_path} "
            f"from local ({float(default_local_pos[i, 0]):.4f}, {float(default_local_pos[i, 1]):.4f}, {float(default_local_pos[i, 2]):.4f}) "
            f"to local ({float(new_local_pos[i, 0]):.4f}, {float(new_local_pos[i, 1]):.4f}, {float(new_local_pos[i, 2]):.4f}) "
            f"with offset ({float(offsets[i, 0]):.4f}, {float(offsets[i, 1]):.4f}, {float(offsets[i, 2]):.4f})"
        )


@configclass
class EventCfg:
    """Configuration for reset events."""

    init_ffw_bg2_pose = EventTerm(
        func=set_default_joint_pose,
        mode="reset",
        params={
            "joint_positions": {
                "arm_r_joint1": 0.638451,
                "arm_r_joint2": -0.924706,
                "arm_r_joint3": -0.737492,
                "arm_r_joint4": -2.572742,
                "arm_r_joint5": -1.302020,
                "arm_r_joint6": 0.350000,
                "arm_r_joint7": -0.527065,
                "head_joint1": 0.694821,
                "head_joint2": -0.350000,
            },
        },
    )

    randomize_led_target_anchor_pose = EventTerm(
        func=randomize_led_target_anchor_pose,
        mode="reset",
        params={
            "offset_range": {
                # Keep local y fixed so the anchor stays on the authored rag-body face.
                "x": (-0.12, 0.12),
                "y": (0.0, 0.0),
                "z": (-0.08, 0.08),
            },
            "led_anchor_prim_expr": "/World/envs/env_.*/DummyRag/LedTargetAnchor",
            "rag_body_prim_expr": "/World/envs/env_.*/DummyRag",
            "asset_cfg": SceneEntityCfg("dummy_rag"),
        },
    )


@configclass
class FFWBG2BaseEnvCfg(BaseEnvCfg):
    """FFW_BG2 base environment: robot standing in an empty room."""

    def __post_init__(self):
        super().__post_init__()

        self.events = EventCfg()

        # Set FFW_BG2 as robot
        self.scene.robot = FFW_BG2_WITHOUT_MIMIC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        arm_joint_names = [
            "arm_r_joint1", "arm_r_joint2", "arm_r_joint3", "arm_r_joint4",
            "arm_r_joint5", "arm_r_joint6", "arm_r_joint7",
        ]
        head_joint_names = ["head_joint1", "head_joint2"]
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            name="robot", joint_names=arm_joint_names + head_joint_names
        )
        self.observations.policy.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            name="robot", joint_names=arm_joint_names + head_joint_names
        )

        # IK-based arm action (orientation locked: rotation scale = 0)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["arm_r_joint[1-7]"],
            body_name="arm_r_link7",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=(0.3, 0.3, 0.3, 0.0, 0.0, 0.0),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
        )

        # Gripper action
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper_r_joint[1-4]"],
            open_command_expr={"gripper_r_joint.*": 0.0},
            close_command_expr={"gripper_r_joint.*": 1.0},
        )

        # Relative head action keeps the current pose when no key is pressed.
        self.actions.head_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=head_joint_names,
            scale=0.05,
        )

        # Right wrist camera
        self.scene.right_wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ffw_bg2_follower/right_arm/arm_r_link7/camera_r_bottom_screw_frame/camera_r_link/right_wrist_cam",
            update_period=0.0,
            height=244,
            width=244,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(-0.08, 0.0, 0.0), rot=(0.5, -0.5, -0.5, 0.5), convention="isaac"
            ),
        )

        # Head camera
        self.scene.head_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ffw_bg2_follower/head/head_link2/head_cam",
            update_period=0.0,
            height=244,
            width=244,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=12.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(-0.03, 0.04, 0.0), rot=(0.5, 0.5, -0.5, -0.5), convention="isaac"
            ),
        )

        # End-effector frame transformer
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ffw_bg2_follower/world",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ffw_bg2_follower/right_arm/arm_r_link7",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
                ),
            ],
        )
