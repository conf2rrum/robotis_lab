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

import os
import threading
import torch
import cv2
from pynput.keyboard import Listener
from collections.abc import Callable
from datetime import datetime

from robotis_dds_python.idl.trajectory_msgs.msg import JointTrajectory_
from robotis_dds_python.idl.sensor_msgs.msg import JointState_
from robotis_dds_python.idl.sensor_msgs.msg import CompressedImage_
from robotis_dds_python.idl.std_msgs.msg import Header_
from robotis_dds_python.idl.builtin_interfaces.msg import Time_

from robotis_dds_python.tools.topic_manager import TopicManager


class OMYSdk:
    """OMYSdk class for DDS teleoperation and publishing robot state/images."""

    def __init__(self, env, mode: str):
        self.env = env
        self.mode = mode  # 'record' or 'inference'
        self.running = True
        self.domain_id = int(os.getenv("ROS_DOMAIN_ID", 0))
        self.joint_trajectory_cmd = None
        self._started = False
        self._reset_state = False
        self._additional_callbacks = {}
        self.lock = threading.Lock()  # Protect shared state
        
        # Initialize current joint state - will be updated only when commands are received
        self.current_joint_state = {}

        # Define joint names
        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "rh_r1_joint"]
        self.exclude_joints = []

        # DDS Topic Manager
        topic_manager = TopicManager(domain_id=self.domain_id)

        # Subscribers
        self.joint_trajectory_reader = topic_manager.topic_reader(
            topic_name="leader/joint_trajectory",
            topic_type=JointTrajectory_
        )

        # Publishers
        self.joint_state_writer = topic_manager.topic_writer(
            topic_name="joint_states",
            topic_type=JointState_
        )
        self.top_cam_writer = topic_manager.topic_writer(
            topic_name="camera/cam_top/color/image_rect_raw/compressed",
            topic_type=CompressedImage_
        )
        self.wrist_cam_writer = topic_manager.topic_writer(
            topic_name="camera/cam_wrist/color/image_rect_raw/compressed",
            topic_type=CompressedImage_
        )

        # Start subscriber thread
        self.thread = threading.Thread(target=self._subscriber_loop, daemon=True)
        self.thread.start()

        # Keyboard listener
        self.listener = Listener(on_press=self._on_press)
        self.listener.start()

        self._keyboard_controls()

    # ----------------------
    # Keyboard controls
    # ----------------------
    def _keyboard_controls(self):
        print("\n[Control] Press keys to control the robot:")
        if self.mode == 'record':
            print("[N] Save successful episode and proceed to the next one")
            print("[R] Skip failed episode (not saved) and proceed to the next one")
            print("[B] Start recording the current episode")
        elif self.mode == 'inference':
            print("[R] Skip failed episode (not saved) and proceed to the next one")
            print("[B] Start/Resume robot control")

    def _on_press(self, key):
        try:
            if self.mode == 'record':
                if key.char == 'b':
                    self._started = True
                    self._reset_state = False
                elif key.char == 'r':
                    self._started = False
                    self._reset_state = True
                    self._call_callback("R")
                elif key.char == 'n':
                    self._started = False
                    self._reset_state = True
                    self._call_callback("N")
            elif self.mode == 'inference':
                if key.char == 'b':
                    self._started = True
                    self._reset_state = False
                elif key.char == 'r':
                    self._started = False
                    self._reset_state = True
                    self._call_callback("R")
        except AttributeError:
            pass

    def _call_callback(self, key):
        if key in self._additional_callbacks:
            self._additional_callbacks[key]()

    # ----------------------
    # Subscriber loop
    # ----------------------
    def _subscriber_loop(self):
        """Continuously read joint trajectory commands from the leader."""
        try:
            while self.running:
                for msg in self.joint_trajectory_reader.take_iter():
                    if msg and msg.points:
                        joint_dict = dict(zip(msg.joint_names, msg.points[-1].positions))
                        with self.lock:
                            self.joint_trajectory_cmd = [
                                joint_dict.get(name, 0.0) for name in self.joint_names
                            ]
        except Exception as e:
            print("Subscriber thread exception:", e)
        finally:
            try:
                self.joint_trajectory_reader.Close()
            except:
                pass
            print("Subscriber closed")

    # ----------------------
    # Publishers
    # ----------------------
    def _publish_joint_states(self):
        """Publish current joint states over DDS."""
        now = datetime.now()
        stamp = Time_(sec=int(now.timestamp()), nanosec=now.microsecond * 1000)
        header = Header_(stamp=stamp, frame_id="base_link")

        obs_joint_name = self.env.scene["robot"].data.joint_names
        all_positions = self.env.scene["robot"].data.joint_pos.squeeze(0).tolist()
        all_velocities = self.env.scene["robot"].data.joint_vel.squeeze(0).tolist()
        all_efforts = [0.0] * len(all_positions)

        # Flatten nested lists if necessary
        if isinstance(all_positions[0], list):
            all_positions = [p for sub in all_positions for p in sub]
        if isinstance(all_velocities[0], list):
            all_velocities = [v for sub in all_velocities for v in sub]

        # Get indices of the joints we care about
        indices = [obs_joint_name.index(name) for name in self.joint_names]

        positions = [all_positions[i] for i in indices]
        velocities = [all_velocities[i] for i in indices]
        efforts = [all_efforts[i] for i in indices]

        joint_state = JointState_(
            header=header,
            name=list(self.joint_names),
            position=positions,
            velocity=velocities,
            effort=efforts
        )

        try:
            self.joint_state_writer.write(joint_state)
        except Exception as e:
            print("[Writer] write error:", e)

    def _publish_camera(self, cam_name: str):
        """Publish camera image as DDS compressed image."""
        try:
            cam_data = self.env.scene[cam_name].data
            img = cam_data.output['rgb'][0].cpu().numpy()  # Convert tensor to numpy

            _, buffer = cv2.imencode('.jpg', img)
            jpeg_bytes = buffer.tobytes()

            now = datetime.now()
            stamp = Time_(sec=int(now.timestamp()), nanosec=now.microsecond * 1000)
            header = Header_(stamp=stamp, frame_id="camera_frame")

            msg = CompressedImage_(header=header, format="jpeg", data=jpeg_bytes)
            if cam_name == "cam_wrist":
                self.wrist_cam_writer.write(msg)
            elif cam_name == "cam_top":
                self.top_cam_writer.write(msg)
        except Exception as e:
            print("Camera publish error:", e)

    # ----------------------
    # Action/state handling
    # ----------------------
    def _compute_action_state(self):
        """Compute current action dictionary based on keyboard input and subscriber."""
        state = {'reset': self._reset_state, 'started': self._started}
        if state['reset']:
            self._reset_state = False
            return state
        state['joint_state'] = self._get_device_state()
        return state

    def _get_device_state(self):
        """Return latest joint positions, starting with current robot state and updating with received commands."""
        with self.lock:
            # Start with current robot joint positions
            obs_joint_name = self.env.scene["robot"].data.joint_names
            all_positions = self.env.scene["robot"].data.joint_pos.squeeze(0).tolist()
            
            # Flatten nested lists if necessary
            if isinstance(all_positions[0], list):
                all_positions = [p for sub in all_positions for p in sub]

            # Build joint state from current robot state
            joint_state = {}
            for name in self.joint_names:
                if name in obs_joint_name:
                    idx = obs_joint_name.index(name)
                    joint_state[name] = all_positions[idx]
                else:
                    joint_state[name] = 0.0  # Fallback only if joint not found in robot
            
            # Update with received commands if available
            if self.joint_trajectory_cmd:
                cmd_dict = dict(zip(self.joint_names, self.joint_trajectory_cmd))
                joint_state.update(cmd_dict)
            
            return joint_state

    def get_action(self):
        """Return action tensor for robot control."""
        action = self._compute_action_state()
        if action['reset']:
            return {"reset": True}
        if not action['started']:
            return None

        joint_state = action['joint_state']
        positions = [joint_state.get(name, 0.0) for name in self.joint_names]
        return torch.tensor(positions, device=self.env.device, dtype=torch.float32).unsqueeze(0)

    def publish_observations(self):
        """Publish joint states and camera images."""
        self._publish_joint_states()
        self._publish_camera("cam_top")
        self._publish_camera("cam_wrist")

    # ----------------------
    # Utility
    # ----------------------
    def shutdown(self):
        """Stop threads and close DDS publishers/subscribers."""
        self.running = False
        self.thread.join()
        for obj in [self.joint_trajectory_reader, self.joint_state_writer,
                    self.top_cam_writer, self.wrist_cam_writer]:
            try:
                obj.Close()
            except:
                pass
        print("OMYSdk shutdown complete")

    def reset(self):
        self._reset_state = False

    def add_callback(self, key: str, func: Callable):
        """Add callback function for a specific key."""
        self._additional_callbacks[key] = func
