"""FFW_SG2 base environment: empty room with robot and keyboard teleop."""

import gymnasium as gym
import os

from . import agents
from .ffw_sg2_base_env_cfg import FFWSG2BaseEnvCfg
from .mimic_env import FFWSG2BaseMimicEnv
from .mimic_env_cfg import FFWSG2BaseMimicEnvCfg

gym.register(
    id="RobotisLab-Base-FFW-SG2-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": FFWSG2BaseEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_image.json"),
    },
    disable_env_checker=True,
)

gym.register(
    id="RobotisLab-Base-FFW-SG2-Mimic-v0",
    entry_point="robotis_lab.simulation_tasks.manager_based.FFW_SG2.base:FFWSG2BaseMimicEnv",
    kwargs={
        "env_cfg_entry_point": FFWSG2BaseMimicEnvCfg,
    },
    disable_env_checker=True,
)
