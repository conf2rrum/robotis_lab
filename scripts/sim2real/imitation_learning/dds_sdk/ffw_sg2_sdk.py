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
import time
from pynput.keyboard import Listener
from collections.abc import Callable
from datetime import datetime

from robotis_dds_python.idl.trajectory_msgs.msg import JointTrajectory_
from robotis_dds_python.idl.sensor_msgs.msg import JointState_
from robotis_dds_python.idl.sensor_msgs.msg import CompressedImage_
from robotis_dds_python.idl.std_msgs.msg import Header_
from robotis_dds_python.idl.std_msgs.msg import String_
from robotis_dds_python.idl.builtin_interfaces.msg import Time_

from robotis_dds_python.tools.topic_manager import TopicManager


class FFWSG2Sdk:
    """FFWSG2Sdk class for DDS teleoperation and publishing humanoid robot state/images."""

    def __init__(self, env, mode: str):
        self.env = env
        self.mode = mode  # 'record' or 'inference'
        self.running = True
        self.domain_id = int(os.getenv("ROS_DOMAIN_ID", 0))
        self.left_arm_trajectory_cmd = None
        self.right_arm_trajectory_cmd = None
        self._started = False
        self._reset_state = False
        self._additional_callbacks = {}
        self._first_episode = True  # Track if this is the first episode
        self._episode_phase = "idle"  # Current state: "idle" (waiting) or "recording" (active episode)
        self.lock = threading.Lock()  # Protect shared state
        
        # Initialize current joint state - will be updated only when commands are received
        self.current_joint_state = {}

        # Define joint names for FFW_SG2 humanoid robot
        self.joint_names = [
            "arm_l_joint1", "arm_l_joint2", "arm_l_joint3", "arm_l_joint4", "arm_l_joint5", "arm_l_joint6", "arm_l_joint7", "gripper_l_joint1",
            "arm_r_joint1", "arm_r_joint2", "arm_r_joint3", "arm_r_joint4", "arm_r_joint5", "arm_r_joint6", "arm_r_joint7", "gripper_r_joint1",
            "lift_joint", "head_joint1", "head_joint2"
        ]

        # DDS Topic Manager
        topic_manager = TopicManager(domain_id=self.domain_id)

        # Subscribers for both arms
        self.left_arm_joint_trajectory_reader = topic_manager.topic_reader(
            topic_name="/leader/joint_trajectory_command_broadcaster_left/joint_trajectory",
            topic_type=JointTrajectory_
        )
        self.right_arm_joint_trajectory_reader = topic_manager.topic_reader(
            topic_name="/leader/joint_trajectory_command_broadcaster_right/joint_trajectory",
            topic_type=JointTrajectory_
        )
        self.lift_joint_trajectory_reader = topic_manager.topic_reader(
            topic_name="/leader/joystick_controller_right/joint_trajectory",
            topic_type=JointTrajectory_
        )
        self.joystick_track_trigger_reader = topic_manager.topic_reader(
            topic_name="/leader/joystick_controller/tact_trigger",
            topic_type=String_
        )

        # Publishers
        self.joint_state_writer = topic_manager.topic_writer(
            topic_name="joint_states",
            topic_type=JointState_
        )
        self.head_cam_writer = topic_manager.topic_writer(
            topic_name="/zed/zed_node/left/image_rect_color/compressed",
            topic_type=CompressedImage_
        )
        self.right_wrist_cam_writer = topic_manager.topic_writer(
            topic_name="/camera_right/camera_right/color/image_rect_raw/compressed",
            topic_type=CompressedImage_
        )
        self.left_wrist_cam_writer = topic_manager.topic_writer(
            topic_name="/camera_left/camera_left/color/image_rect_raw/compressed",
            topic_type=CompressedImage_
        )

        # Start subscriber threads for both arms
        self.left_thread = threading.Thread(target=self._left_arm_subscriber_loop, daemon=True)
        self.right_thread = threading.Thread(target=self._right_arm_subscriber_loop, daemon=True)
        self.lift_thread = threading.Thread(target=self._lift_joint_subscriber_loop, daemon=True)
        self.joystick_thread = threading.Thread(target=self._joystick_subscriber_loop, daemon=True)

        self.left_thread.start()
        self.right_thread.start()
        self.lift_thread.start()
        self.joystick_thread.start()

        # Keyboard listener
        self.listener = Listener(on_press=self._on_press)
        self.listener.start()

        self._keyboard_controls()

    # ----------------------
    # Keyboard controls
    # ----------------------
    def _keyboard_controls(self):
        print("\n[Control] Press keys to control the FFW_SG2 robot:")
        if self.mode == 'record':
            print("[N] Save successful episode and proceed to the next one")
            print("[R] Skip failed episode (not saved) and proceed to the next one")
            print("[B] Start recording the current episode")
        elif self.mode == 'inference':
            print("[N] Save successful episode and proceed to the next one")
            print("[R] Skip failed episode (not saved) and proceed to the next one")
            print("[B] Start/Resume robot control")

    def _on_press(self, key):
        try:
            if self.mode == 'record':
                if key.char == 'b':
                    self._started = True
                    self._reset_state = False
                    # Update episode tracking when manually starting
                    if self._first_episode:
                        self._first_episode = False
                    self._episode_phase = "recording"  # Now recording
                elif key.char == 'r':
                    self._started = False
                    self._reset_state = True
                    self._call_callback("R")
                    # If resetting while recording before first episode was saved, go back to first episode state
                    if self._episode_phase == "recording" and not self._first_episode:
                        self._first_episode = True
                        self._episode_phase = "idle"
                elif key.char == 'n':
                    self._started = False
                    self._reset_state = True
                    self._call_callback("N")
                    # After saving, go back to idle state
                    self._episode_phase = "idle"
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
    # Subscriber loops for both arms
    # ----------------------
    def _left_arm_subscriber_loop(self):
        """Continuously read joint trajectory commands from the leader."""
        try:
            while self.running:
                for msg in self.left_arm_joint_trajectory_reader.take_iter():
                    if msg and msg.points:
                        joint_dict = dict(zip(msg.joint_names, msg.points[-1].positions))
                        with self.lock:
                            self.left_arm_trajectory_cmd = joint_dict
                time.sleep(0.001)  # 1ms sleep to reduce CPU load
        except Exception as e:
            print("Left arm subscriber thread exception:", e)
        finally:
            try:
                self.left_arm_joint_trajectory_reader.Close()
            except:
                pass
            print("Left arm subscriber closed")

    def _right_arm_subscriber_loop(self):
        """Continuously read right arm joint trajectory commands from the leader."""
        try:
            while self.running:
                for msg in self.right_arm_joint_trajectory_reader.take_iter():
                    if msg and msg.points:
                        joint_dict = dict(zip(msg.joint_names, msg.points[-1].positions))
                        with self.lock:
                            self.right_arm_trajectory_cmd = joint_dict
                time.sleep(0.001)  # 1ms sleep to reduce CPU load
        except Exception as e:
            print("Right arm subscriber thread exception:", e)
        finally:
            try:
                self.right_arm_joint_trajectory_reader.Close()
            except:
                pass
            print("Right arm subscriber closed")

    def _lift_joint_subscriber_loop(self):
        """Continuously read lift joint trajectory commands from the leader."""
        try:
            while self.running:
                for msg in self.lift_joint_trajectory_reader.take_iter():
                    if msg and msg.points:
                        joint_dict = dict(zip(msg.joint_names, msg.points[-1].positions))
                        with self.lock:
                            # Update only the lift joint command
                            self.left_arm_trajectory_cmd = self.left_arm_trajectory_cmd or {}
                            self.left_arm_trajectory_cmd.update(joint_dict)
                time.sleep(0.001)  # 1ms sleep to reduce CPU load
        except Exception as e:
            print("Lift joint subscriber thread exception:", e)
        finally:
            try:
                self.lift_joint_trajectory_reader.Close()
            except:
                pass
            print("Lift joint subscriber closed")

    def _joystick_subscriber_loop(self):
        """Continuously read joystick track trigger commands from the leader."""
        try:
            while self.running:
                for msg in self.joystick_track_trigger_reader.take_iter():
                    # Only process joystick triggers in record mode
                    if self.mode != 'record':
                        continue

                    tack_trigger = msg.data
                    if tack_trigger == 'right':
                        with self.lock:
                            if self._first_episode:
                                # First episode: only start recording
                                self._started = True
                                self._reset_state = False
                                self._first_episode = False
                                self._episode_phase = "recording"  # Now recording
                            elif self._episode_phase == "recording":
                                # Currently recording: save episode and go back to idle
                                self._started = False
                                self._reset_state = True
                                self._call_callback("N")
                                self._episode_phase = "idle"  # Now idle, waiting for next start
                            elif self._episode_phase == "idle":
                                # Currently idle: start new episode
                                self._started = True
                                self._reset_state = False
                                self._episode_phase = "recording"  # Now recording
                    elif tack_trigger == 'left':
                        with self.lock:
                            # Reset current episode (don't save)
                            self._started = False
                            self._reset_state = True
                            self._call_callback("R")
                            # If resetting while recording before first episode was saved, go back to first episode state
                            if self._episode_phase == "recording" and not self._first_episode:
                                # We started recording but haven't saved yet - reset to first episode
                                self._first_episode = True
                                self._episode_phase = "idle"
                time.sleep(0.001)  # 1ms sleep to reduce CPU load
        except Exception as e:
            print("Joystick subscriber thread exception:", e)
        finally:
            try:
                self.joystick_track_trigger_reader.Close()
            except:
                pass
            print("Joystick subscriber closed")

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
        indices = [obs_joint_name.index(name) for name in self.joint_names if name in obs_joint_name]

        positions = [all_positions[i] for i in indices]
        velocities = [all_velocities[i] for i in indices]
        efforts = [all_efforts[i] for i in indices]

        joint_state = JointState_(
            header=header,
            name=[self.joint_names[i] for i in range(len(indices))],
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
            img = cam_data.output['rgb'][0].cpu().numpy()  # Convert tensor to numpy (RGB format)
            
            # Convert RGB to BGR for OpenCV encoding
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            _, buffer = cv2.imencode('.jpg', img_bgr)
            jpeg_bytes = buffer.tobytes()

            now = datetime.now()
            stamp = Time_(sec=int(now.timestamp()), nanosec=now.microsecond * 1000)
            header = Header_(stamp=stamp, frame_id="camera_frame")

            msg = CompressedImage_(header=header, format="jpeg", data=jpeg_bytes)
            
            # Map camera names to publishers for FFW_SG2
            if cam_name == "cam_wrist_right":
                self.right_wrist_cam_writer.write(msg)
            elif cam_name == "cam_wrist_left":
                self.left_wrist_cam_writer.write(msg)
            elif cam_name == "cam_head":
                self.head_cam_writer.write(msg)
        except Exception as e:
            print(f"Camera publish error for {cam_name}:", e)

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
            
            # Update with left arm commands if available
            if self.left_arm_trajectory_cmd:
                joint_state.update(self.left_arm_trajectory_cmd)
            
            # Update with right arm commands if available
            if self.right_arm_trajectory_cmd:
                joint_state.update(self.right_arm_trajectory_cmd)
            
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
        self._publish_camera("cam_head")
        # self._publish_camera("cam_wrist_right")
        # self._publish_camera("cam_wrist_left")

    # ----------------------
    # Utility
    # ----------------------
    def shutdown(self):
        """Stop threads and close DDS publishers/subscribers."""
        self.running = False
        self.left_thread.join()
        self.right_thread.join()
        
        for obj in [self.left_arm_joint_trajectory_reader, self.right_arm_joint_trajectory_reader,
                    self.joint_state_writer, self.head_cam_writer, 
                    self.right_wrist_cam_writer, self.left_wrist_cam_writer]:
            try:
                obj.Close()
            except:
                pass
        print("FFWSG2Sdk shutdown complete")

    def reset(self):
        self._reset_state = False

    def add_callback(self, key: str, func: Callable):
        """Add callback function for a specific key."""
        self._additional_callbacks[key] = func
