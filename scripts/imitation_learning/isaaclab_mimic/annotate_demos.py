# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Script to add mimic annotations to demos to be used as source demos for mimic dataset generation.
"""

import argparse
import math

from isaaclab.app import AppLauncher

# Launching Isaac Sim Simulator first.


# add argparse arguments
parser = argparse.ArgumentParser(description="Annotate demonstrations for Isaac Lab environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--input_file", type=str, default="./datasets/dataset.hdf5", help="File name of the dataset to be annotated."
)
parser.add_argument(
    "--output_file",
    type=str,
    default="./datasets/dataset_annotated.hdf5",
    help="File name of the annotated output dataset file.",
)
parser.add_argument("--auto", action="store_true", default=False, help="Automatically annotate subtasks.")
parser.add_argument(
    "--success_check_window",
    type=int,
    default=5,
    help=(
        "Accept replay success if the success term becomes true within the final N replay actions. "
        "Use 1 to require success on the final replay state only."
    ),
)
parser.add_argument(
    "--arm_rotation_scale",
    type=float,
    default=None,
    help=(
        "IK rotation action scale to use while replaying demos. If omitted, FFW base tasks use the teleop script"
        " default of 0.35 and other tasks keep their environment default."
    ),
)
parser.add_argument(
    "--head_action_scale",
    type=float,
    default=None,
    help="Head joint delta scale to use while replaying demos. If omitted, the environment default is kept.",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
parser.add_argument(
    "--annotate_subtask_start_signals",
    action="store_true",
    default=False,
    help="Enable annotating start points of subtasks.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and not the one installed by Isaac Sim
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import gymnasium as gym
import os
import torch

import isaaclab_mimic.envs  # noqa: F401

if args_cli.enable_pinocchio:
    import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401

# Only enables inputs if this script is NOT headless mode
if not args_cli.headless and not os.environ.get("HEADLESS", 0):
    from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg

from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import RecorderTerm, RecorderTermCfg, TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import robotis_lab  # noqa: F401
from robotis_lab.simulation_tasks.manager_based.FFW_BG2.base.led_target_anchor_state import (
    LedTargetAnchorInitialStateRecorderCfg,
    LedTargetAnchorPostStepStatesRecorderCfg,
    get_episode_led_target_anchor_root_pose,
    get_led_target_anchor_root_pose,
    queue_led_target_anchor_restore_from_episode,
    restore_led_target_anchor_from_episode,
)

is_paused = False
current_action_index = 0
marked_subtask_action_indices = []
skip_episode = False


def evaluate_success_term(env: ManagerBasedRLMimicEnv, success_term: TerminationTermCfg | None) -> bool:
    """Evaluate a single-env success term."""

    if success_term is None:
        return True
    return bool(success_term.func(env, **success_term.params)[0])


def describe_led_success_state(env: ManagerBasedRLMimicEnv, success_term: TerminationTermCfg | None) -> str | None:
    """Return a compact camera LED diagnostic for FFW base success checks."""

    if success_term is None:
        return None

    params = getattr(success_term, "params", {}) or {}
    sensor_cfg = params.get("sensor_cfg")
    if sensor_cfg is None or not hasattr(sensor_cfg, "name") or sensor_cfg.name not in env.scene.keys():
        return None

    camera = env.scene[sensor_cfg.name]
    rgb = camera.data.output.get("rgb")
    if rgb is None:
        return f"{sensor_cfg.name}: no rgb output"

    rgb = rgb[..., :3]
    _, height, width, _ = rgb.shape
    red_min = params.get("red_min", 180)
    green_max = params.get("green_max", 80)
    blue_max = params.get("blue_max", 80)
    center_ratio = params.get("center_ratio", 0.2)
    min_red_pixels = params.get("min_red_pixels", 50)
    min_center_coverage = params.get("min_center_coverage", 0.8)

    is_red = (rgb[:, :, :, 0] >= red_min) & (rgb[:, :, :, 1] <= green_max) & (rgb[:, :, :, 2] <= blue_max)
    total_red = int(is_red[0].sum().item())
    margin_h = int(height * (1.0 - center_ratio) / 2.0)
    margin_w = int(width * (1.0 - center_ratio) / 2.0)
    center_red = is_red[:, margin_h : height - margin_h, margin_w : width - margin_w]
    center_red_count = int(center_red[0].sum().item())

    if total_red == 0:
        return (
            f"{sensor_cfg.name}: red_pixels=0, required>={min_red_pixels}, "
            f"center_window=x[{margin_w},{width - margin_w}) y[{margin_h},{height - margin_h})"
        )

    red_weights = is_red.to(dtype=torch.float32)
    y_coords = torch.arange(height, device=rgb.device, dtype=torch.float32).view(1, height, 1)
    x_coords = torch.arange(width, device=rgb.device, dtype=torch.float32).view(1, 1, width)
    total_weight = red_weights.sum(dim=(1, 2)).clamp_min(1.0)
    centroid_y = float(((red_weights * y_coords).sum(dim=(1, 2)) / total_weight)[0].item())
    centroid_x = float(((red_weights * x_coords).sum(dim=(1, 2)) / total_weight)[0].item())
    center_coverage = center_red_count / float(total_red)
    return (
        f"{sensor_cfg.name}: red_pixels={total_red}, centroid=({centroid_x:.1f}, {centroid_y:.1f}), "
        f"center_coverage={center_coverage:.3f}, required_pixels>={min_red_pixels}, "
        f"required_coverage>={min_center_coverage:.3f}, "
        f"center_window=x[{margin_w},{width - margin_w}) y[{margin_h},{height - margin_h})"
    )


def print_led_target_anchor_restore_check(
    env: ManagerBasedRLMimicEnv,
    expected_anchor_pose: torch.Tensor | None,
) -> None:
    """Print the actual restored LedTargetAnchor pose and its error from the episode pose."""

    actual_anchor_pose = get_led_target_anchor_root_pose(env, [0])
    if actual_anchor_pose is None:
        print("\tWARNING: LedTargetAnchor stage pose could not be read after restore.")
        return

    actual_anchor_pose = torch.as_tensor(actual_anchor_pose).reshape(-1, 7)[0]
    actual_anchor_pos = actual_anchor_pose[:3].detach().cpu()
    message = (
        "\tVerified LedTargetAnchor stage pose: "
        f"pos=({actual_anchor_pos[0]:.4f}, {actual_anchor_pos[1]:.4f}, {actual_anchor_pos[2]:.4f})"
    )
    if expected_anchor_pose is not None:
        expected_anchor_pose = torch.as_tensor(expected_anchor_pose, device=actual_anchor_pose.device).reshape(-1, 7)[0]
        pos_error = torch.linalg.norm(actual_anchor_pose[:3] - expected_anchor_pose[:3]).item()
        quat_error = torch.linalg.norm(actual_anchor_pose[3:7] - expected_anchor_pose[3:7]).item()
        message += f", pos_error={pos_error:.6f}, quat_error={quat_error:.6f}"
    print(message)


def print_keyboard_event(key: str, status: str):
    """Print the latest keyboard event with the current annotation state."""

    print(
        f'[Keyboard] "{key}" pressed -> {status} '
        f"(paused={is_paused}, skip_episode={skip_episode}, action_index={current_action_index}, "
        f"marked_signals={len(marked_subtask_action_indices)})"
    )


def play_cb():
    global is_paused
    is_paused = False
    print_keyboard_event("N", "replay running")


def pause_cb():
    global is_paused
    is_paused = True
    print_keyboard_event("B", "replay paused")


def skip_episode_cb():
    global skip_episode
    skip_episode = True
    print_keyboard_event("Q", "current episode will be skipped")


def mark_subtask_cb():
    global current_action_index, marked_subtask_action_indices
    marked_subtask_action_indices.append(current_action_index)
    print_keyboard_event("S", f"subtask signal marked at action index {current_action_index}")


def print_keyboard_shortcuts(auto_mode: bool):
    """Print the keyboard shortcuts available during annotation replay."""

    print("\nKeyboard shortcuts:")
    print('  "N": begin or resume replay')
    print('  "B": pause replay')
    print('  "Q": skip the current episode')
    if not auto_mode:
        print('  "S": mark a subtask signal at the current action index')
    print("")


class PreStepDatagenInfoRecorder(RecorderTerm):
    """Recorder term that records the datagen info data in each step."""

    def record_pre_step(self):
        eef_pose_dict = {}
        for eef_name in self._env.cfg.subtask_configs.keys():
            eef_pose_dict[eef_name] = self._env.get_robot_eef_pose(eef_name=eef_name)

        datagen_info = {
            "object_pose": self._env.get_object_poses(),
            "eef_pose": eef_pose_dict,
            "target_eef_pose": self._env.action_to_target_eef_pose(self._env.action_manager.action),
        }
        return "obs/datagen_info", datagen_info


@configclass
class PreStepDatagenInfoRecorderCfg(RecorderTermCfg):
    """Configuration for the datagen info recorder term."""

    class_type: type[RecorderTerm] = PreStepDatagenInfoRecorder


class PreStepSubtaskStartsObservationsRecorder(RecorderTerm):
    """Recorder term that records the subtask start observations in each step."""

    def record_pre_step(self):
        return "obs/datagen_info/subtask_start_signals", self._env.get_subtask_start_signals()


@configclass
class PreStepSubtaskStartsObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the subtask start observations recorder term."""

    class_type: type[RecorderTerm] = PreStepSubtaskStartsObservationsRecorder


class PreStepSubtaskTermsObservationsRecorder(RecorderTerm):
    """Recorder term that records the subtask completion observations in each step."""

    def record_pre_step(self):
        return "obs/datagen_info/subtask_term_signals", self._env.get_subtask_term_signals()


@configclass
class PreStepSubtaskTermsObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the step subtask terms observation recorder term."""

    class_type: type[RecorderTerm] = PreStepSubtaskTermsObservationsRecorder


@configclass
class MimicRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Mimic specific recorder terms."""

    record_initial_state = LedTargetAnchorInitialStateRecorderCfg()
    record_post_step_states = LedTargetAnchorPostStepStatesRecorderCfg()
    record_pre_step_datagen_info = PreStepDatagenInfoRecorderCfg()
    record_pre_step_subtask_start_signals = PreStepSubtaskStartsObservationsRecorderCfg()
    record_pre_step_subtask_term_signals = PreStepSubtaskTermsObservationsRecorderCfg()


def main():
    """Add Isaac Lab Mimic annotations to the given demo dataset file."""
    global is_paused, current_action_index, marked_subtask_action_indices

    # Load input dataset to be annotated
    if not os.path.exists(args_cli.input_file):
        raise FileNotFoundError(f"The input dataset file {args_cli.input_file} does not exist.")
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.input_file)
    dataset_env_name = dataset_file_handler.get_env_name()
    env_name = dataset_env_name
    episode_count = dataset_file_handler.get_num_episodes()

    if episode_count == 0:
        print("No episodes found in the dataset.")
        return 0

    # get output directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.output_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.output_file))[0]
    # create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args_cli.task is not None:
        env_name = args_cli.task.split(":")[-1]
        if dataset_env_name is not None and env_name != dataset_env_name:
            print(
                "[Annotate] WARNING: input dataset was recorded with env "
                f"'{dataset_env_name}', but replay will use --task env '{env_name}'. "
                "Camera mounts and robot assets must match for image-space LED replay."
            )
    if env_name is None:
        raise ValueError("Task/env name was not specified nor found in the dataset.")
    print(f"[Annotate] Input dataset env: {dataset_env_name}; replay env: {env_name}; episodes: {episode_count}.")

    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=1)

    env_cfg.env_name = env_name

    if getattr(env_cfg, "events", None) is not None and hasattr(env_cfg.events, "randomize_led_target_anchor_pose"):
        env_cfg.events.randomize_led_target_anchor_pose = None
        print("[Annotate] Disabled reset-time LedTargetAnchor randomization; replay restores recorded poses.")

    is_ffw_base_task = "Base-FFW" in env_name
    if is_ffw_base_task and not getattr(args_cli, "enable_cameras", False):
        print("[Annotate] WARNING: --enable_cameras is not set; camera-based LED success checks may fail.")

    arm_rotation_scale = args_cli.arm_rotation_scale
    if arm_rotation_scale is None and is_ffw_base_task and hasattr(env_cfg.actions, "arm_action"):
        arm_rotation_scale = 0.35
    if arm_rotation_scale is not None and hasattr(env_cfg.actions, "arm_action"):
        env_cfg.actions.arm_action.scale = (
            0.3,
            0.3,
            0.3,
            arm_rotation_scale,
            arm_rotation_scale,
            arm_rotation_scale,
        )
        print(f"[Annotate] Using arm action scale {env_cfg.actions.arm_action.scale}.")

    if args_cli.head_action_scale is not None and hasattr(env_cfg.actions, "head_action"):
        env_cfg.actions.head_action.scale = args_cli.head_action_scale
        print(f"[Annotate] Using head action scale {env_cfg.actions.head_action.scale}.")

    # extract success checking function to invoke manually
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        raise NotImplementedError("No success termination term was found in the environment.")

    # Disable all termination terms
    env_cfg.terminations = None

    # Set up recorder terms for mimic annotations
    env_cfg.recorders = MimicRecorderManagerCfg()
    if not args_cli.auto:
        # disable subtask term signals recorder term if in manual mode
        env_cfg.recorders.record_pre_step_subtask_term_signals = None

    if not args_cli.auto or (args_cli.auto and not args_cli.annotate_subtask_start_signals):
        # disable subtask start signals recorder term if in manual mode or no need for subtask start annotations
        env_cfg.recorders.record_pre_step_subtask_start_signals = None

    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name

    # create environment from loaded config
    env: ManagerBasedRLMimicEnv = gym.make(env_name, cfg=env_cfg).unwrapped

    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise ValueError("The environment should be derived from ManagerBasedRLMimicEnv")

    if args_cli.auto:
        # check if the mimic API env.get_subtask_term_signals() is implemented
        if env.get_subtask_term_signals.__func__ is ManagerBasedRLMimicEnv.get_subtask_term_signals:
            raise NotImplementedError(
                "The environment does not implement the get_subtask_term_signals method required "
                "to run automatic annotations."
            )
        if (
            args_cli.annotate_subtask_start_signals
            and env.get_subtask_start_signals.__func__ is ManagerBasedRLMimicEnv.get_subtask_start_signals
        ):
            raise NotImplementedError(
                "The environment does not implement the get_subtask_start_signals method required "
                "to run automatic annotations."
            )
    else:
        # get subtask termination signal names for each eef from the environment configs
        subtask_term_signal_names = {}
        subtask_start_signal_names = {}
        for eef_name, eef_subtask_configs in env.cfg.subtask_configs.items():
            subtask_start_signal_names[eef_name] = (
                [subtask_config.subtask_term_signal for subtask_config in eef_subtask_configs]
                if args_cli.annotate_subtask_start_signals
                else []
            )
            subtask_term_signal_names[eef_name] = [
                subtask_config.subtask_term_signal for subtask_config in eef_subtask_configs
            ]
            # Validation: if annotating start signals, every subtask (including the last) must have a name
            if args_cli.annotate_subtask_start_signals:
                if any(name in (None, "") for name in subtask_start_signal_names[eef_name]):
                    raise ValueError(
                        f"Missing 'subtask_term_signal' for one or more subtasks in eef '{eef_name}'. When"
                        " '--annotate_subtask_start_signals' is enabled, each subtask (including the last) must"
                        " specify 'subtask_term_signal'. The last subtask's term signal name is used as the final"
                        " start signal name."
                    )
            # no need to annotate the last subtask term signal, so remove it from the list
            subtask_term_signal_names[eef_name].pop()

    # reset environment
    env.reset()

    # Only enables inputs if this script is NOT headless mode
    if not args_cli.headless and not os.environ.get("HEADLESS", 0):
        keyboard_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.1, rot_sensitivity=0.1))
        keyboard_interface.add_callback("N", play_cb)
        keyboard_interface.add_callback("B", pause_cb)
        keyboard_interface.add_callback("Q", skip_episode_cb)
        if not args_cli.auto:
            keyboard_interface.add_callback("S", mark_subtask_cb)
        keyboard_interface.reset()
        print_keyboard_shortcuts(auto_mode=args_cli.auto)

    # simulate environment -- run everything in inference mode
    exported_episode_count = 0
    processed_episode_count = 0
    successful_task_count = 0  # Counter for successful task completions
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():
            # Iterate over the episodes in the loaded dataset file
            for episode_index, episode_name in enumerate(dataset_file_handler.get_episode_names()):
                processed_episode_count += 1
                print(f"\nAnnotating episode #{episode_index} ({episode_name})")
                episode = dataset_file_handler.load_episode(episode_name, env.device)

                is_episode_annotated_successfully = False
                if args_cli.auto:
                    is_episode_annotated_successfully = annotate_episode_in_auto_mode(env, episode, success_term)
                else:
                    is_episode_annotated_successfully = annotate_episode_in_manual_mode(
                        env, episode, success_term, subtask_term_signal_names, subtask_start_signal_names
                    )

                if is_episode_annotated_successfully and not skip_episode:
                    # set success to the recorded episode data and export to file
                    env.recorder_manager.set_success_to_episodes(
                        None, torch.tensor([[True]], dtype=torch.bool, device=env.device)
                    )
                    env.recorder_manager.export_episodes()
                    exported_episode_count += 1
                    successful_task_count += 1  # Increment successful task counter
                    print("\tExported the annotated episode.")
                else:
                    print("\tSkipped exporting the episode due to incomplete subtask annotations.")
            break

    print(
        f"\nExported {exported_episode_count} (out of {processed_episode_count}) annotated"
        f" episode{'s' if exported_episode_count > 1 else ''}."
    )
    print(
        f"Successful task completions: {successful_task_count}"
    )  # This line is used by the dataset generation test case to check if the expected number of demos were annotated
    print("Exiting the app.")

    # Close environment after annotation is complete
    env.close()

    return successful_task_count


def replay_episode(
    env: ManagerBasedRLMimicEnv,
    episode: EpisodeData,
    success_term: TerminationTermCfg | None = None,
) -> bool:
    """Replays an episode in the environment.

    This function replays the given recorded episode in the environment. It can optionally check if the task
    was successfully completed using a success termination condition input.

    Args:
        env: The environment to replay the episode in.
        episode: The recorded episode data to replay.
        success_term: Optional termination term to check for task success.

    Returns:
        True if the episode was successfully replayed and the success condition was met (if provided),
        False otherwise.
    """
    global current_action_index, skip_episode, is_paused
    # read initial state and actions from the loaded episode
    initial_state = episode.data["initial_state"]
    actions = episode.data["actions"]
    env.sim.reset()
    env.recorder_manager.reset()
    queue_led_target_anchor_restore_from_episode(env, episode.data)
    env.reset_to(initial_state, None, is_relative=True)
    restored_anchor_source = restore_led_target_anchor_from_episode(env, episode.data)
    if restored_anchor_source is not None:
        restored_anchor_pose, _ = get_episode_led_target_anchor_root_pose(episode.data)
        if restored_anchor_pose is not None:
            restored_anchor_pose = torch.as_tensor(restored_anchor_pose).reshape(-1, 7)[0]
            restored_anchor_pos = restored_anchor_pose[:3].detach().cpu().tolist()
            print(
                "\tRestored LedTargetAnchor from "
                f"{restored_anchor_source}: pos=({restored_anchor_pos[0]:.4f}, "
                f"{restored_anchor_pos[1]:.4f}, {restored_anchor_pos[2]:.4f})"
            )
        env.sim.forward()
        if env.sim.has_rtx_sensors():
            env.sim.render()
        print_led_target_anchor_restore_check(env, restored_anchor_pose)
    else:
        print("\tWARNING: LedTargetAnchor pose was not restored from this episode.")
    success_action_indices = []
    for action_index, action in enumerate(actions):
        current_action_index = action_index
        while is_paused or skip_episode:
            env.sim.render()
            if skip_episode:
                return False
            continue
        action_tensor = torch.Tensor(action).reshape([1, action.shape[0]])
        env.step(torch.Tensor(action_tensor))
        if evaluate_success_term(env, success_term):
            success_action_indices.append(action_index)
    if success_term is not None:
        success_check_window = max(1, args_cli.success_check_window)
        final_action_index = len(actions) - 1
        final_success = success_action_indices and success_action_indices[-1] == final_action_index
        recent_success = success_action_indices and success_action_indices[-1] >= final_action_index - success_check_window + 1
        if not final_success and recent_success:
            print(
                "\tSuccess was detected near the end of replay "
                f"(action_index={success_action_indices[-1]}, window={success_check_window}); accepting episode."
            )
        elif not final_success:
            led_debug = describe_led_success_state(env, success_term)
            if led_debug is not None:
                print(f"\tFinal success check failed: {led_debug}")
            return False
    return True


def annotate_episode_in_auto_mode(
    env: ManagerBasedRLMimicEnv,
    episode: EpisodeData,
    success_term: TerminationTermCfg | None = None,
) -> bool:
    """Annotates an episode in automatic mode.

    This function replays the given episode in the environment and checks if the task was successfully completed.
    If the task was not completed, it will print a message and return False. Otherwise, it will check if all the
    subtask term signals are annotated and return True if they are, False otherwise.

    Args:
        env: The environment to replay the episode in.
        episode: The recorded episode data to replay.
        success_term: Optional termination term to check for task success.

    Returns:
        True if the episode was successfully annotated, False otherwise.
    """
    global skip_episode
    skip_episode = False
    is_episode_annotated_successfully = replay_episode(env, episode, success_term)
    if skip_episode:
        print("\tSkipping the episode.")
        return False
    if not is_episode_annotated_successfully:
        print("\tThe final task was not completed.")
    else:
        # check if all the subtask term signals are annotated
        annotated_episode = env.recorder_manager.get_episode(0)
        subtask_term_signal_dict = annotated_episode.data["obs"]["datagen_info"]["subtask_term_signals"]
        for signal_name, signal_flags in subtask_term_signal_dict.items():
            signal_flags = torch.tensor(signal_flags, device=env.device)
            if not torch.any(signal_flags):
                is_episode_annotated_successfully = False
                print(f'\tDid not detect completion for the subtask "{signal_name}".')
        if args_cli.annotate_subtask_start_signals:
            subtask_start_signal_dict = annotated_episode.data["obs"]["datagen_info"]["subtask_start_signals"]
            for signal_name, signal_flags in subtask_start_signal_dict.items():
                if not torch.any(signal_flags):
                    is_episode_annotated_successfully = False
                    print(f'\tDid not detect start for the subtask "{signal_name}".')
    return is_episode_annotated_successfully


def annotate_episode_in_manual_mode(
    env: ManagerBasedRLMimicEnv,
    episode: EpisodeData,
    success_term: TerminationTermCfg | None = None,
    subtask_term_signal_names: dict[str, list[str]] = {},
    subtask_start_signal_names: dict[str, list[str]] = {},
) -> bool:
    """Annotates an episode in manual mode.

    This function replays the given episode in the environment and allows for manual marking of subtask term signals.
    It iterates over each eef and prompts the user to mark the subtask term signals for that eef.

    Args:
        env: The environment to replay the episode in.
        episode: The recorded episode data to replay.
        success_term: Optional termination term to check for task success.
        subtask_term_signal_names: Dictionary mapping eef names to lists of subtask term signal names.
        subtask_start_signal_names: Dictionary mapping eef names to lists of subtask start signal names.
    Returns:
        True if the episode was successfully annotated, False otherwise.
    """
    global is_paused, marked_subtask_action_indices, skip_episode
    # iterate over the eefs for marking subtask term signals
    subtask_term_signal_action_indices = {}
    subtask_start_signal_action_indices = {}
    for eef_name, eef_subtask_term_signal_names in subtask_term_signal_names.items():
        eef_subtask_start_signal_names = subtask_start_signal_names[eef_name]
        # skip if no subtask annotation is needed for this eef
        if len(eef_subtask_term_signal_names) == 0 and len(eef_subtask_start_signal_names) == 0:
            continue

        while True:
            is_paused = True
            skip_episode = False
            print(f'\tPlaying the episode for subtask annotations for eef "{eef_name}".')
            print("\tSubtask signals to annotate:")
            if len(eef_subtask_start_signal_names) > 0:
                print(f"\t\t- Start:\t{eef_subtask_start_signal_names}")
            print(f"\t\t- Termination:\t{eef_subtask_term_signal_names}")

            print('\n\tPress "N" to begin.')
            print('\tPress "B" to pause.')
            print('\tPress "S" to annotate subtask signals.')
            print('\tPress "Q" to skip the episode.\n')
            marked_subtask_action_indices = []
            task_success_result = replay_episode(env, episode, success_term)
            if skip_episode:
                print("\tSkipping the episode.")
                return False

            print(f"\tSubtasks marked at action indices: {marked_subtask_action_indices}")
            expected_subtask_signal_count = len(eef_subtask_term_signal_names) + len(eef_subtask_start_signal_names)
            if task_success_result and expected_subtask_signal_count == len(marked_subtask_action_indices):
                print(f'\tAll {expected_subtask_signal_count} subtask signals for eef "{eef_name}" were annotated.')
                for marked_signal_index in range(expected_subtask_signal_count):
                    if args_cli.annotate_subtask_start_signals and marked_signal_index % 2 == 0:
                        subtask_start_signal_action_indices[
                            eef_subtask_start_signal_names[int(marked_signal_index / 2)]
                        ] = marked_subtask_action_indices[marked_signal_index]
                    if not args_cli.annotate_subtask_start_signals:
                        # Direct mapping when only collecting termination signals
                        subtask_term_signal_action_indices[eef_subtask_term_signal_names[marked_signal_index]] = (
                            marked_subtask_action_indices[marked_signal_index]
                        )
                    elif args_cli.annotate_subtask_start_signals and marked_signal_index % 2 == 1:
                        # Every other signal is a termination when collecting both types
                        subtask_term_signal_action_indices[
                            eef_subtask_term_signal_names[math.floor(marked_signal_index / 2)]
                        ] = marked_subtask_action_indices[marked_signal_index]
                break

            if not task_success_result:
                print("\tThe final task was not completed.")
                return False

            if expected_subtask_signal_count != len(marked_subtask_action_indices):
                print(
                    f"\tOnly {len(marked_subtask_action_indices)} out of"
                    f' {expected_subtask_signal_count} subtask signals for eef "{eef_name}" were'
                    " annotated."
                )

            print(f'\tThe episode will be replayed again for re-marking subtask signals for the eef "{eef_name}".\n')

    annotated_episode = env.recorder_manager.get_episode(0)
    for (
        subtask_term_signal_name,
        subtask_term_signal_action_index,
    ) in subtask_term_signal_action_indices.items():
        # subtask termination signal is false until subtask is complete, and true afterwards
        subtask_signals = torch.ones(len(episode.data["actions"]), dtype=torch.bool)
        subtask_signals[:subtask_term_signal_action_index] = False
        annotated_episode.add(f"obs/datagen_info/subtask_term_signals/{subtask_term_signal_name}", subtask_signals)

    if args_cli.annotate_subtask_start_signals:
        for (
            subtask_start_signal_name,
            subtask_start_signal_action_index,
        ) in subtask_start_signal_action_indices.items():
            subtask_signals = torch.ones(len(episode.data["actions"]), dtype=torch.bool)
            subtask_signals[:subtask_start_signal_action_index] = False
            annotated_episode.add(
                f"obs/datagen_info/subtask_start_signals/{subtask_start_signal_name}", subtask_signals
            )

    return True


if __name__ == "__main__":
    # run the main function
    successful_task_count = main()
    # close sim app
    simulation_app.close()
    # exit with the number of successful task completions as return code
    exit(successful_task_count)
