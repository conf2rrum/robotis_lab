"""Keyboard teleoperation for FFW_BG2 right arm, gripper, and head joints."""

from __future__ import annotations

import weakref
from collections.abc import Callable

import carb
import numpy as np
import omni
import torch
from scipy.spatial.transform import Rotation

from isaaclab.devices import DeviceBase


class FFWBG2ArmHeadKeyboard(DeviceBase):
    """Keyboard controller for right-arm SE(3), gripper, and 2-DoF head control.

    The output action layout matches the FFW_BG2 base environment:
    ``[arm_delta_pose(6), gripper(1), head_delta(2)]``.
    By default, arm rotation keys are disabled so the wrist camera stays parallel to the wall.
    """

    def __init__(
        self,
        pos_sensitivity: float = 0.05,
        rot_sensitivity: float = 0.05,
        head_sensitivity: float = 1.0,
        enable_arm_rotation: bool = False,
        sim_device: str = "cpu",
    ):
        super().__init__()

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.head_sensitivity = head_sensitivity
        self.enable_arm_rotation = enable_arm_rotation
        self._sim_device = sim_device

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )

        self._additional_callbacks: dict[str, Callable] = {}
        self._create_key_bindings()
        self.reset()

    def __del__(self):
        """Release the keyboard subscription."""
        if getattr(self, "_keyboard_sub", None) is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def __str__(self) -> str:
        lines = [
            "FFW_BG2 Arm+Head Keyboard",
            "\tReset key state: L",
            "\tReset environment: R",
            "\tPrint current joint positions: J",
            "\tToggle gripper: K",
            "\tArm position: W/S, A/D, Q/E",
            "\tHead pan: U/O",
            "\tHead tilt: I/P",
        ]
        if self.enable_arm_rotation:
            lines.insert(5, "\tArm rotation: Z/X, T/G, C/V")
        else:
            lines.append("\tArm rotation: locked to keep the camera parallel to the wall")
        return "\n".join(lines)

    def reset(self):
        self._close_gripper = False
        self._delta_pos = np.zeros(3)
        self._delta_rot = np.zeros(3)
        self._head_delta = np.zeros(2)

    def add_callback(self, key: str, func: Callable):
        self._additional_callbacks[key] = func

    def advance(self) -> torch.Tensor:
        rot_vec = Rotation.from_euler("XYZ", self._delta_rot).as_rotvec()
        gripper_value = -1.0 if self._close_gripper else 1.0
        command = np.concatenate([self._delta_pos, rot_vec, np.asarray([gripper_value]), self._head_delta])
        return torch.tensor(command, dtype=torch.float32, device=self._sim_device)

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "L":
                self.reset()
            elif event.input.name == "K":
                self._close_gripper = not self._close_gripper
            elif event.input.name in self._ARM_POS_KEYS:
                self._delta_pos += self._INPUT_KEY_MAPPING[event.input.name]
            elif self.enable_arm_rotation and event.input.name in self._ARM_ROT_KEYS:
                self._delta_rot += self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in self._HEAD_KEYS:
                self._head_delta += self._INPUT_KEY_MAPPING[event.input.name]

        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._ARM_POS_KEYS:
                self._delta_pos -= self._INPUT_KEY_MAPPING[event.input.name]
            elif self.enable_arm_rotation and event.input.name in self._ARM_ROT_KEYS:
                self._delta_rot -= self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in self._HEAD_KEYS:
                self._head_delta -= self._INPUT_KEY_MAPPING[event.input.name]

        if event.type == carb.input.KeyboardEventType.KEY_PRESS and event.input.name in self._additional_callbacks:
            self._additional_callbacks[event.input.name]()

        return True

    def _create_key_bindings(self):
        self._ARM_POS_KEYS = {"W", "S", "A", "D", "Q", "E"}
        self._ARM_ROT_KEYS = {"Z", "X", "T", "G", "C", "V"}
        self._HEAD_KEYS = {"U", "O", "I", "P"}
        self._INPUT_KEY_MAPPING = {
            "W": np.asarray([1.0, 0.0, 0.0]) * self.pos_sensitivity,
            "S": np.asarray([-1.0, 0.0, 0.0]) * self.pos_sensitivity,
            "A": np.asarray([0.0, 1.0, 0.0]) * self.pos_sensitivity,
            "D": np.asarray([0.0, -1.0, 0.0]) * self.pos_sensitivity,
            "Q": np.asarray([0.0, 0.0, 1.0]) * self.pos_sensitivity,
            "E": np.asarray([0.0, 0.0, -1.0]) * self.pos_sensitivity,
            "Z": np.asarray([1.0, 0.0, 0.0]) * self.rot_sensitivity,
            "X": np.asarray([-1.0, 0.0, 0.0]) * self.rot_sensitivity,
            "T": np.asarray([0.0, 1.0, 0.0]) * self.rot_sensitivity,
            "G": np.asarray([0.0, -1.0, 0.0]) * self.rot_sensitivity,
            "C": np.asarray([0.0, 0.0, 1.0]) * self.rot_sensitivity,
            "V": np.asarray([0.0, 0.0, -1.0]) * self.rot_sensitivity,
            "U": np.asarray([1.0, 0.0]) * self.head_sensitivity,
            "O": np.asarray([-1.0, 0.0]) * self.head_sensitivity,
            "I": np.asarray([0.0, 1.0]) * self.head_sensitivity,
            "P": np.asarray([0.0, -1.0]) * self.head_sensitivity,
        }
