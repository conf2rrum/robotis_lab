"""FFW_SG2-specific base environment: robot in an empty room with IK control."""

import isaaclab.sim as sim_utils
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from robotis_lab.assets.robots.FFW_SG2 import FFW_SG2_CFG  # isort: skip
from robotis_lab.simulation_tasks.manager_based.FFW_BG2.base import mdp
from robotis_lab.simulation_tasks.manager_based.FFW_BG2.base.ffw_bg2_base_env_cfg import (
    randomize_led_target_anchor_pose,
    set_default_joint_pose,
)

from .base_env_cfg import BaseEnvCfg


@configclass
class EventCfg:
    """Configuration for reset events."""

    init_ffw_sg2_pose = EventTerm(
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
                "lift_joint": 0.0,
            },
        },
    )

    randomize_led_target_anchor_pose = EventTerm(
        func=randomize_led_target_anchor_pose,
        mode="reset",
        params={
            "offset_range": {
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
class FFWSG2BaseEnvCfg(BaseEnvCfg):
    """FFW_SG2 base environment: robot standing in an empty room."""

    def __post_init__(self):
        super().__post_init__()

        self.events = EventCfg()

        self.scene.robot = FFW_SG2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        arm_joint_names = [
            "arm_r_joint1",
            "arm_r_joint2",
            "arm_r_joint3",
            "arm_r_joint4",
            "arm_r_joint5",
            "arm_r_joint6",
            "arm_r_joint7",
        ]
        head_joint_names = ["head_joint1", "head_joint2"]
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            name="robot", joint_names=arm_joint_names + head_joint_names
        )
        self.observations.policy.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            name="robot", joint_names=arm_joint_names + head_joint_names
        )

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["arm_r_joint[1-7]"],
            body_name="arm_r_link7",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=(0.3, 0.3, 0.3, 0.0, 0.0, 0.0),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
        )

        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper_r_joint[1-4]"],
            open_command_expr={"gripper_r_joint.*": 0.0},
            close_command_expr={"gripper_r_joint.*": 1.0},
        )

        self.actions.head_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=head_joint_names,
            scale=0.05,
        )

        self.scene.right_wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ffw_sg2_follower/arm_r_link7/camera_r_bottom_screw_frame/camera_r_link/right_wrist_cam",
            update_period=0.0,
            height=244,
            width=244,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 2),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(-0.08, 0.0, 0.0),
                # rot=(0.5, -0.5, -0.5, 0.5),
                rot=(0.0, -0.7071, 0.0, 0.7071),
                convention="isaac",
            ),
        )

        self.scene.head_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ffw_sg2_follower/head_link2/zed/cam_head",
            update_period=0.0,
            height=376,
            width=672,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=10.4,
                focus_distance=200.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 100.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, 0.03, 0.0),
                rot=(0.5, 0.5, -0.5, -0.5),
                convention="isaac",
            ),
        )

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ffw_sg2_follower/arm_base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ffw_sg2_follower/arm_r_link7",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
                ),
            ],
        )
