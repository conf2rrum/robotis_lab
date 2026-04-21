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

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.managers import SceneEntityCfg

from robotis_lab.simulation_tasks.manager_based.FFW_BG2.base import mdp


LED_TARGET_ANCHOR_EXPR = "/World/envs/env_.*/DummyRag/LedTargetAnchor"
SEARCH_DONE_SIGNAL_NAME = "search_done"
SUCCESS_SIGNAL_NAME = "led_centered"
START_SIGNAL_NAME = "search_started"
AUXILIARY_ACTION_SLICE = slice(6, 9)
ARM_MOTION_POS_THRESHOLD = 0.02
ARM_MOTION_ROT_THRESHOLD = 0.12
SUCCESS_TERM_KWARGS = {
    "sensor_cfg": SceneEntityCfg("right_wrist_cam"),
    "center_ratio": 0.35,
    "red_min": 180,
    "green_max": 80,
    "blue_max": 80,
    "min_red_pixels": 50,
    "min_center_coverage": 0.8,
}


class FFWSG2BaseMimicEnv(ManagerBasedRLMimicEnv):
    """Isaac Lab Mimic environment wrapper for the FFW_SG2 base LED-search task."""

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = slice(None)

        eef_pos = self.obs_buf["policy"]["eef_pos"][env_ids]
        eef_quat = self.obs_buf["policy"]["eef_quat"][env_ids]
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        (target_eef_pose,) = target_eef_pose_dict.values()
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)

        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=[env_id])[0]
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        delta_position = target_pos - curr_pos

        delta_rot_mat = target_rot.matmul(curr_rot.transpose(-1, -2))
        delta_quat = PoseUtils.quat_from_matrix(delta_rot_mat)
        delta_rotation = PoseUtils.axis_angle_from_quat(delta_quat)

        (auxiliary_action,) = gripper_action_dict.values()
        pose_action = torch.cat([delta_position, delta_rotation], dim=0)
        if action_noise_dict is not None:
            noise = action_noise_dict[eef_name] * torch.randn_like(pose_action)
            pose_action += noise
            pose_action = torch.clamp(pose_action, -1.0, 1.0)

        return torch.cat([pose_action, auxiliary_action.reshape(-1)], dim=0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        delta_position = action[:, :3]
        delta_rotation = action[:, 3:6]

        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=None)
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        target_pos = curr_pos + delta_position

        delta_rotation_angle = torch.linalg.norm(delta_rotation, dim=-1, keepdim=True)
        delta_rotation_axis = delta_rotation / delta_rotation_angle

        is_close_to_zero_angle = torch.isclose(delta_rotation_angle, torch.zeros_like(delta_rotation_angle)).squeeze(1)
        delta_rotation_axis[is_close_to_zero_angle] = torch.zeros_like(delta_rotation_axis)[is_close_to_zero_angle]

        delta_quat = PoseUtils.quat_from_angle_axis(delta_rotation_angle.squeeze(1), delta_rotation_axis).squeeze(0)
        delta_rot_mat = PoseUtils.matrix_from_quat(delta_quat)
        target_rot = torch.matmul(delta_rot_mat, curr_rot)

        target_poses = PoseUtils.make_pose(target_pos, target_rot).clone()
        return {eef_name: target_poses}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        return {list(self.cfg.subtask_configs.keys())[0]: actions[..., AUXILIARY_ACTION_SLICE]}

    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)

        object_pose_matrix = super().get_object_poses(env_ids=env_ids)
        led_target_anchor_pose = self._get_led_target_anchor_pose(env_ids=env_ids)
        if led_target_anchor_pose is not None:
            object_pose_matrix["led_target_anchor"] = led_target_anchor_pose
        elif "dummy_rag" in object_pose_matrix:
            if not getattr(self, "_warned_missing_led_target_anchor", False):
                print(
                    "[FFWSG2BaseMimicEnv] LedTargetAnchor was not found in the stage; "
                    "falling back to the DummyRag root pose for object-centric transforms."
                )
                self._warned_missing_led_target_anchor = True
            object_pose_matrix["led_target_anchor"] = object_pose_matrix["dummy_rag"]

        return object_pose_matrix

    def get_subtask_start_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)

        started = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        return {START_SIGNAL_NAME: started[env_ids]}

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)

        search_done = self._get_search_done_signals()
        success = mdp.red_led_in_center(self, **SUCCESS_TERM_KWARGS)
        return {
            SEARCH_DONE_SIGNAL_NAME: search_done[env_ids],
            SUCCESS_SIGNAL_NAME: success[env_ids],
        }

    def _get_search_done_signals(self) -> torch.Tensor:
        eef_name = list(self.cfg.subtask_configs.keys())[0]
        curr_pose = self.get_robot_eef_pose(eef_name=eef_name, env_ids=None)
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        if (
            not hasattr(self, "_search_reference_pos")
            or self._search_reference_pos.shape != curr_pos.shape
            or self._search_reference_pos.device != curr_pos.device
        ):
            self._search_reference_pos = curr_pos.clone()
            self._search_reference_rot = curr_rot.clone()
            self._search_done_signals = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        reset_mask = self.episode_length_buf <= 1
        if torch.any(reset_mask):
            self._search_reference_pos[reset_mask] = curr_pos[reset_mask]
            self._search_reference_rot[reset_mask] = curr_rot[reset_mask]
            self._search_done_signals[reset_mask] = False

        pos_delta = torch.linalg.norm(curr_pos - self._search_reference_pos, dim=-1)
        delta_rot_mat = curr_rot.matmul(self._search_reference_rot.transpose(-1, -2))
        delta_quat = PoseUtils.quat_from_matrix(delta_rot_mat)
        rot_delta = torch.linalg.norm(PoseUtils.axis_angle_from_quat(delta_quat), dim=-1)

        arm_motion_started = (pos_delta > ARM_MOTION_POS_THRESHOLD) | (rot_delta > ARM_MOTION_ROT_THRESHOLD)
        self._search_done_signals |= arm_motion_started
        return self._search_done_signals

    def _get_led_target_anchor_pose(self, env_ids: Sequence[int] | None = None) -> torch.Tensor | None:
        self._ensure_led_target_anchor_view()

        if getattr(self, "_led_target_anchor_view", None) is None:
            return None

        if env_ids is None:
            env_ids = slice(None)

        anchor_pos_w, anchor_quat_w = self._led_target_anchor_view.get_world_poses()
        anchor_pos = anchor_pos_w[env_ids] - self.scene.env_origins[env_ids, :3]
        anchor_quat = anchor_quat_w[env_ids]
        return PoseUtils.make_pose(anchor_pos, PoseUtils.matrix_from_quat(anchor_quat))

    def _ensure_led_target_anchor_view(self) -> None:
        if getattr(self, "_led_target_anchor_view", None) is not None:
            return

        from isaacsim.core.prims import XFormPrim

        candidate_exprs = [LED_TARGET_ANCHOR_EXPR]
        if "dummy_rag" in self.scene.keys():
            asset_root_expr = self.scene["dummy_rag"].cfg.prim_path.rstrip("/")
            candidate_exprs.append(f"{asset_root_expr}/LedTargetAnchor")

        for expr in candidate_exprs:
            if not expr:
                continue

            prim_paths = sim_utils.find_matching_prim_paths(expr)
            if prim_paths:
                self._led_target_anchor_view = XFormPrim(expr, reset_xform_properties=False)
                self._led_target_anchor_expr = expr
                self._led_target_anchor_prim_paths = prim_paths
                return
