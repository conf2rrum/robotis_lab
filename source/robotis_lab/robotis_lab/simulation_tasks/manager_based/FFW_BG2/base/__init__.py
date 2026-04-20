"""FFW_BG2 base environment: empty room with robot and keyboard teleop."""

import gymnasium as gym
import os

from . import agents
from .ffw_bg2_base_env_cfg import FFWBG2BaseEnvCfg
from .mimic_env import FFWBG2BaseMimicEnv
from .mimic_env_cfg import FFWBG2BaseMimicEnvCfg

gym.register(
    id="RobotisLab-Base-FFW-BG2-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": FFWBG2BaseEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_image.json"),
    },
    disable_env_checker=True,
)

gym.register(
    id="RobotisLab-Base-FFW-BG2-Mimic-v0",
    entry_point="robotis_lab.simulation_tasks.manager_based.FFW_BG2.base:FFWBG2BaseMimicEnv",
    kwargs={
        "env_cfg_entry_point": FFWBG2BaseMimicEnvCfg,
    },
    disable_env_checker=True,
)
