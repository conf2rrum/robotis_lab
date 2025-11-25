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

import h5py
import numpy as np
import argparse
from tqdm import tqdm
from datetime import datetime

from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROBOT_CONFIGS = {
    "OMY": {
        "expected_dim": 7,
        "joint_names": [
            "joint1.pos", "joint2.pos", "joint3.pos", "joint4.pos",
            "joint5.pos", "joint6.pos", "rh_r1_joint.pos",
        ],
        "cameras": {
            "cam_wrist": {"height": 480, "width": 848},
            "cam_top": {"height": 480, "width": 848},
        }
    },
    "FFW_SG2": {
        "expected_dim": 19,
        "joint_names": [
            "arm_l_joint1.pos", "arm_l_joint2.pos", "arm_l_joint3.pos", "arm_l_joint4.pos",
            "arm_l_joint5.pos", "arm_l_joint6.pos", "arm_l_joint7.pos",
            "arm_r_joint1.pos", "arm_r_joint2.pos", "arm_r_joint3.pos", "arm_r_joint4.pos",
            "arm_r_joint5.pos", "arm_r_joint6.pos", "arm_r_joint7.pos",
            "gripper_l_joint1.pos", "gripper_r_joint1.pos",
            "head_joint1.pos", "head_joint2.pos", "lift_joint.pos",
        ],
        "cameras": {
            "cam_head_left": {"height": 376, "width": 672},
        }
    }
}

def get_env_features(fps: int, robot_type: str):
    if robot_type not in ROBOT_CONFIGS:
        raise ValueError(f"Unsupported robot type: {robot_type}")
    
    config = ROBOT_CONFIGS[robot_type]
    
    # Build action and observation.state features
    features = {
        "action": {
            "dtype": "float32",
            "shape": (config["expected_dim"],),
            "names": config["joint_names"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (config["expected_dim"],),
            "names": config["joint_names"],
        }
    }
    
    # Add camera features
    for cam_name, cam_cfg in config["cameras"].items():
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video",
            "shape": [cam_cfg["height"], cam_cfg["width"], 3],
            "names": ["height", "width", "channels"],
            "video_info": {
                "video.height": cam_cfg["height"],
                "video.width": cam_cfg["width"],
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": float(fps),
                "video.channels": 3,
                "has_audio": False,
            },
        }
    
    return features

def process_data(dataset: LeRobotDataset, task: str, demo_group: h5py.Group, demo_name: str, frame_skip: int, robot_type: str) -> bool:
    """
    Process a single demonstration group from the HDF5 dataset
    and add it into the LeRobot dataset.
    """
    if robot_type not in ROBOT_CONFIGS:
        raise ValueError(f"Unsupported robot type: {robot_type}")
    
    config = ROBOT_CONFIGS[robot_type]
    camera_keys = list(config["cameras"].keys())
    
    try:
        # Load action and joint position data
        actions = np.array(demo_group['actions'], dtype=np.float32)
        joint_pos = np.array(demo_group['obs/joint_pos'], dtype=np.float32)
        
        # Load camera images based on robot type
        camera_data = {}
        for cam_key in camera_keys:
            camera_data[cam_key] = np.array(demo_group[f'obs/{cam_key}'], dtype=np.uint8)
            
    except KeyError as e:
        print(f"Demo {demo_name} is not valid (missing key: {e}), skipping...")
        return False

    if actions.shape[0] < 10:
        print(f"Demo {demo_name} has insufficient frames ({actions.shape[0]}), skipping...")
        return False

    # Ensure actions and joint positions are 2D arrays
    if actions.ndim == 1:
        actions = actions.reshape(-1, config["expected_dim"])
    if joint_pos.ndim == 1:
        joint_pos = joint_pos.reshape(-1, config["expected_dim"])
    
    total_state_frames = actions.shape[0]

    # Process each frame
    for frame_index in tqdm(range(total_state_frames), desc=f"Processing demo {demo_name}"):
        if frame_index < frame_skip:
            continue
        
        # Build frame dictionary
        frame = {
            "action": actions[frame_index],
            "observation.state": joint_pos[frame_index],
        }
        
        # Add camera images
        for cam_key in camera_keys:
            frame[f"observation.images.{cam_key}"] = camera_data[cam_key][frame_index]
        
        dataset.add_frame(frame=frame, task=task)

    return True

def convert_isaaclab_to_lerobot(
    task: str, repo_id: str, robot_type: str, dataset_file: str,
    fps: int, push_to_hub: bool = False, frame_skip: int = 3, root: str = "./datasets/lerobot/sim2real_data"
):
    """
    Convert an IsaacLab HDF5 dataset into LeRobot dataset format.
    """
    hdf5_files = [dataset_file]
    now_episode_index = 0

    # Create a new LeRobot dataset
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=get_env_features(fps, robot_type),
        root=root,
    )

    # Process each HDF5 dataset file
    for hdf5_id, hdf5_file in enumerate(hdf5_files):
        print(f"[{hdf5_id+1}/{len(hdf5_files)}] Processing HDF5 file: {hdf5_file}")
        with h5py.File(hdf5_file, "r") as f:
            demo_names = list(f["data"].keys())
            print(f"Found {len(demo_names)} demos: {demo_names}")

            for demo_name in tqdm(demo_names, desc="Processing each demo"):
                demo_group = f["data"][demo_name]

                # Skip unsuccessful demonstrations
                if "success" in demo_group.attrs and not demo_group.attrs["success"]:
                    print(f"Demo {demo_name} not successful, skipping...")
                    continue

                valid = process_data(dataset, task, demo_group, demo_name, frame_skip, robot_type)

                if valid:
                    now_episode_index += 1
                    dataset.save_episode()
                    print(f"Saved episode {now_episode_index} successfully")

    # Optionally push to HuggingFace Hub
    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert IsaacLab dataset to LeRobot format")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., OMY_Pickup)")
    parser.add_argument("--robot_type", type=str, default="OMY", help="Robot type (default: OMY)")
    parser.add_argument("--dataset_file", type=str, default="./datasets/dataset.hdf5", help="Path to dataset HDF5 file")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for dataset (default: 10)")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push dataset to HuggingFace Hub")
    parser.add_argument("--frame_skip", type=int, default=2, help="Frame skip rate (default: 2)")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    default_repo_id = f"./datasets/lerobot/{timestamp}"
    parser.add_argument("--repo_id", type=str, default=default_repo_id, help=f"Repo ID (default: {default_repo_id})")

    args = parser.parse_args()

    convert_isaaclab_to_lerobot(
        task=args.task,
        repo_id=args.repo_id,
        robot_type=args.robot_type,
        dataset_file=args.dataset_file,
        fps=args.fps,
        push_to_hub=args.push_to_hub,
        frame_skip=args.frame_skip,
        root=default_repo_id,
    )
