# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Taehyeong Kim

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg

from robotis_lab.real_world_tasks.manager_based.FFW_SG2.pick_place.mdp import ffw_sg2_pick_place_events
from robotis_lab.real_world_tasks.manager_based.FFW_SG2.pick_place.pick_place_env_cfg import PickPlaceEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from robotis_lab.assets.robots.FFW_SG2 import FFW_SG2_CFG  # isort: skip
from robotis_lab.assets.object.robotis_net_table import NET_TABLE_CFG
from robotis_lab.assets.object.plastic_basket2 import PLASTIC_BASKET2_CFG
from robotis_lab.assets.object.brush_ring import BRUSH_RING_CFG
from robotis_lab.assets.object.silicone_tube_ring import SILICONE_TUBE_RING_CFG
from robotis_lab.assets.object.pliers_ring import PLIERS_RING_CFG
from robotis_lab.assets.object.scissors_ring import SCISSORS_RING_CFG
from robotis_lab.assets.object.screw_driver_ring import SCREW_DRIVER_RING_CFG

import math


@configclass
class EventCfg:
    """Configuration for events."""

    set_robot_joint_pose = EventTerm(
        func=ffw_sg2_pick_place_events.set_default_joint_pose,
        mode="reset",
        params={
            "joint_positions": {
                "arm_l_joint1": 0.75,
                "arm_l_joint4": -2.30,
                "arm_r_joint1": 0.75,
                "arm_r_joint4": -2.30,
                "head_joint1": 0.549,
                "lift_joint": -0.0993,
            },
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    randomize_ffw_sg2_joint_state = EventTerm(
        func=ffw_sg2_pick_place_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.03,
            "joint_names": ["arm_l_joint1", "arm_l_joint2", "arm_l_joint3", "arm_l_joint4", "arm_l_joint5", "arm_l_joint6", "arm_l_joint7",
                            "arm_r_joint1", "arm_r_joint2", "arm_r_joint3", "arm_r_joint4", "arm_r_joint5", "arm_r_joint6", "arm_r_joint7"],
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    randomize_robot_base_pose = EventTerm(
        func=ffw_sg2_pick_place_events.randomize_robot_base_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    randomize_brush_positions = EventTerm(
        func=ffw_sg2_pick_place_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.55, 0.56), "y": (0.26, 0.26), "z": (0.95, 0.95)},
            "min_separation": 0.1,
            "asset_cfgs": [SceneEntityCfg("brush")],
        },
    )

    randomize_driver_positions = EventTerm(
        func=ffw_sg2_pick_place_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.55, 0.56), "y": (0.087, 0.087), "z": (0.95, 0.95)},
            "min_separation": 0.1,
            "asset_cfgs": [SceneEntityCfg("driver")],
        },
    )

    randomize_silicone_positions = EventTerm(
        func=ffw_sg2_pick_place_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.55, 0.56), "y": (0.26, 0.26), "z": (1.235, 1.235)},
            "min_separation": 0.1,
            "asset_cfgs": [SceneEntityCfg("silicone")],
        },
    )

    randomize_scissors_positions = EventTerm(
        func=ffw_sg2_pick_place_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.55, 0.56), "y": (0.087, 0.087), "z": (1.235, 1.235)},
            "min_separation": 0.1,
            "asset_cfgs": [SceneEntityCfg("scissors")],
        },
    )

    randomize_basket_positions = EventTerm(
        func=ffw_sg2_pick_place_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.41, 0.41), "y": (0.0, 0.0), "z": (0.72, 0.72), "yaw": (0.0, 0.0)},
            "min_separation": 0.1,
            "asset_cfgs": [SceneEntityCfg("basket")],
        },
    )

    set_net_table_position = EventTerm(
        func=ffw_sg2_pick_place_events.set_object_pose,
        mode="reset",
        params={
            "pose": {"x": 0.0, "y": 0.0, "z": 0.0},
            "asset_cfg": SceneEntityCfg("table"),
        },
    )

    randomize_scene_light = EventTerm(
        func=ffw_sg2_pick_place_events.randomize_scene_lighting_domelight,
        mode="reset",
        params={
            "intensity_range": (500.0, 3000.0),
            "color_range": ((0.8, 1.0), (0.8, 1.0), (0.8, 1.0)),
            "asset_cfg": SceneEntityCfg("light"),
        },
    )


@configclass
class FFWSG2PickPlaceEnvCfg(PickPlaceEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set events
        self.events = EventCfg()

        # Set FFWSG2 as robot
        self.scene.robot = FFW_SG2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # Set table
        self.scene.table = NET_TABLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Table")
        self.scene.brush = BRUSH_RING_CFG.replace(prim_path="{ENV_REGEX_NS}/Brush")
        self.scene.silicone = SILICONE_TUBE_RING_CFG.replace(prim_path="{ENV_REGEX_NS}/SiliconeTube")
        self.scene.scissors = SCISSORS_RING_CFG.replace(prim_path="{ENV_REGEX_NS}/Scissors")
        self.scene.driver = SCREW_DRIVER_RING_CFG.replace(prim_path="{ENV_REGEX_NS}/ScrewDriver")
        self.scene.basket = PLASTIC_BASKET2_CFG.replace(prim_path="{ENV_REGEX_NS}/Basket")

        # Add semantics to ground
        self.scene.plane.semantic_tags = [("class", "ground")]

        # self.scene.cam_wrist_right = CameraCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/ffw_sg2_follower/arm_r_link7/camera_r_bottom_screw_frame/camera_r_link/cam_wrist_right",
        #     update_period=0.0,
        #     height=480,
        #     width=848,
        #     data_types=["rgb"],
        #     spawn=sim_utils.PinholeCameraCfg(
        #         focal_length=10.0, focus_distance=200.0, horizontal_aperture=20.955, clipping_range=(0.01, 100.0)
        #     ),
        #     offset=CameraCfg.OffsetCfg(
        #         pos=(0.0, 0.0, 0.0),
        #         rot=(0.5, 0.5, -0.5, -0.5),
        #         convention="isaac",
        #     )
        # )
        # self.scene.cam_wrist_left = CameraCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/ffw_sg2_follower/arm_l_link7/camera_l_bottom_screw_frame/camera_l_link/cam_wrist_left",
        #     update_period=0.0,
        #     height=480,
        #     width=848,
        #     data_types=["rgb"],
        #     spawn=sim_utils.PinholeCameraCfg(
        #         focal_length=10.0, focus_distance=200.0, horizontal_aperture=20.955, clipping_range=(0.01, 100.0)
        #     ),
        #     offset=CameraCfg.OffsetCfg(
        #         pos=(0.0, 0.0, 0.0),
        #         rot=(0.5, 0.5, -0.5, -0.5),
        #         convention="isaac",
        #     )
        # )
        self.scene.cam_head = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ffw_sg2_follower/head_link2/zed/cam_head",
            update_period=0.0,
            height=376,
            width=672,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=10.4, focus_distance=200.0, horizontal_aperture=20.955, clipping_range=(0.01, 100.0)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, 0.03, 0.0),
                rot=(0.5, 0.5, -0.5, -0.5),
                convention="isaac",
            )
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"

        self.scene.right_eef = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ffw_sg2_follower/arm_base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ffw_sg2_follower/arm_r_link7",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, -0.2],
                    ),
                ),
            ],
        )

        self.scene.left_eef = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ffw_sg2_follower/arm_base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ffw_sg2_follower/arm_l_link7",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, -0.2],
                    ),
                ),
            ],
        )
