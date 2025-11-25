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

"""
Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def task_done(env: ManagerBasedRLEnv, brush_cfg: SceneEntityCfg, basket_cfg: SceneEntityCfg, distance_threshold: float = 0.10) -> torch.Tensor:
    """
    Success = brush placed inside the basket.
    """

    brush: RigidObject = env.scene[brush_cfg.name]
    basket: RigidObject = env.scene[basket_cfg.name]

    brush_pos = brush.data.root_pos_w
    basket_pos = basket.data.root_pos_w

    # Check 3D distance between brush and basket
    distance_3d = torch.linalg.vector_norm(brush_pos - basket_pos, dim=1)
    done = distance_3d < distance_threshold

    return done
