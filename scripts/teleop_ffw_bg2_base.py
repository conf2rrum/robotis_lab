"""Keyboard teleoperation for the FFW_SG2 base LED-search task."""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Keyboard teleoperation for FFW_SG2 base arm+head control.")
parser.add_argument("--task", type=str, default="RobotisLab-Base-FFW-SG2-Mimic-v0", help="Task name.")
parser.add_argument("--arm_sensitivity", type=float, default=1.0, help="Sensitivity multiplier for arm control.")
parser.add_argument(
    "--arm_rotation_scale",
    type=float,
    default=0.35,
    help="Scale applied to rotational IK action components.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable Fabric and use USD I/O so Isaac Sim inspectors can read the stage more reliably.",
)
parser.add_argument(
    "--dataset_file",
    type=str,
    default=None,
    help="Optional HDF5 file path to save successful teleop demos.",
)
parser.add_argument(
    "--num_demos",
    type=int,
    default=0,
    help="Number of successful demos to save before exiting. Set to 0 to keep recording until you stop the app.",
)
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=3,
    help="Number of consecutive success steps required before exporting a demo.",
)
parser.add_argument(
    "--fixed_led_target_anchor",
    action="store_true",
    default=False,
    help="Disable LedTargetAnchor randomization so every reset uses the authored fixed LED pose.",
)
parser.add_argument(
    "--debug_led_target_anchor",
    action="store_true",
    default=False,
    help="Print LedTargetAnchor pose after every teleop reset and compare it to the first reset pose.",
)
parser.add_argument("--head_input_sensitivity", type=float, default=1.0, help="Sensitivity multiplier for head input.")
parser.add_argument("--head_action_scale", type=float, default=0.05, help="Per-step head joint delta scale in radians.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import robotis_lab  # noqa: F401
from robotis_lab.devices import FFWBG2ArmHeadKeyboard
from robotis_lab.simulation_tasks.manager_based.FFW_BG2.base.led_target_anchor_state import (
    LedTargetAnchorInitialStateRecorderCfg,
    LedTargetAnchorPostStepStatesRecorderCfg,
    get_led_target_anchor_root_pose,
)
from robotis_lab.simulation_tasks.manager_based.FFW_SG2.base.ffw_sg2_base_env_cfg import (
    randomize_led_target_anchor_pose,
)


LED_TARGET_ANCHOR_OFFSET_RANGE = {
    "x": (-0.12, 0.12),
    "y": (0.0, 0.0),
    "z": (-0.08, 0.08),
}


class PreStepDatagenInfoRecorder(RecorderTerm):
    """Recorder term that captures Mimic datagen information for teleop demos."""

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
    """Configuration for recording Mimic datagen information."""

    class_type: type[RecorderTerm] = PreStepDatagenInfoRecorder


class PreStepSubtaskTermsObservationsRecorder(RecorderTerm):
    """Recorder term that captures Mimic subtask completion signals."""

    def record_pre_step(self):
        return "obs/datagen_info/subtask_term_signals", self._env.get_subtask_term_signals()


@configclass
class PreStepSubtaskTermsObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for recording Mimic subtask completion signals."""

    class_type: type[RecorderTerm] = PreStepSubtaskTermsObservationsRecorder


@configclass
class MimicTeleopRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Recorder configuration for teleop demos that will be reused by Mimic generation."""

    record_initial_state = LedTargetAnchorInitialStateRecorderCfg()
    record_post_step_states = LedTargetAnchorPostStepStatesRecorderCfg()
    record_pre_step_datagen_info = PreStepDatagenInfoRecorderCfg()
    record_pre_step_subtask_term_signals = PreStepSubtaskTermsObservationsRecorderCfg()


def resolve_dataset_path() -> str | None:
    if args_cli.dataset_file:
        return args_cli.dataset_file
    if args_cli.num_demos > 0:
        return "./datasets/ffw_sg2_base_demos.hdf5"
    return None


def setup_dataset_output(dataset_file: str) -> tuple[str, str]:
    output_dir = os.path.dirname(dataset_file) or "."
    output_file_name = os.path.splitext(os.path.basename(dataset_file))[0]
    os.makedirs(output_dir, exist_ok=True)
    return output_dir, output_file_name


def main():
    dataset_file = resolve_dataset_path()
    recording_enabled = dataset_file is not None

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.env_name = args_cli.task.split(":")[-1]
    env_cfg.terminations.time_out = None
    success_term = None
    if hasattr(env_cfg.terminations, "success") and env_cfg.terminations.success is not None:
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None

    if (
        args_cli.fixed_led_target_anchor
        and getattr(env_cfg, "events", None) is not None
        and hasattr(env_cfg.events, "randomize_led_target_anchor_pose")
    ):
        env_cfg.events.randomize_led_target_anchor_pose = None

    mimic_recording_enabled = bool(getattr(env_cfg, "subtask_configs", {}))
    if recording_enabled:
        output_dir, output_file_name = setup_dataset_output(dataset_file)
        env_cfg.recorders = MimicTeleopRecorderManagerCfg() if mimic_recording_enabled else ActionStateRecorderManagerCfg()
        env_cfg.recorders.dataset_export_dir_path = output_dir
        env_cfg.recorders.dataset_filename = output_file_name
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    env_cfg.actions.head_action.scale = args_cli.head_action_scale
    env_cfg.actions.arm_action.scale = (
        0.3,
        0.3,
        0.3,
        args_cli.arm_rotation_scale,
        args_cli.arm_rotation_scale,
        args_cli.arm_rotation_scale,
    )

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    success_step_count = 0
    recorded_demo_count = 0
    total_success_count = 0
    should_reset = False
    led_target_anchor_reference_pose = None

    def request_reset():
        nonlocal should_reset
        should_reset = True

    def print_joint_positions():
        robot = env.scene["robot"]
        joint_names = robot.data.joint_names
        joint_pos = robot.data.joint_pos[0].detach().cpu().tolist()
        target_joint_names = [
            "arm_r_joint1",
            "arm_r_joint2",
            "arm_r_joint3",
            "arm_r_joint4",
            "arm_r_joint5",
            "arm_r_joint6",
            "arm_r_joint7",
            "head_joint1",
            "head_joint2",
        ]

        print("\nCurrent joint positions [rad]:")
        for joint_name in target_joint_names:
            joint_index = joint_names.index(joint_name)
            print(f'    "{joint_name}": {joint_pos[joint_index]:.6f},')
        print("")

    teleop = FFWBG2ArmHeadKeyboard(
        pos_sensitivity=0.05 * args_cli.arm_sensitivity,
        rot_sensitivity=0.05 * args_cli.arm_sensitivity,
        head_sensitivity=args_cli.head_input_sensitivity,
        enable_arm_rotation=True,
        sim_device=env.device,
    )
    teleop.add_callback("R", request_reset)
    teleop.add_callback("J", print_joint_positions)

    reset_terms = set(env.event_manager.active_terms.get("reset", []))
    task_randomizes_led_target_anchor = (
        "randomize_led_target_anchor_pose" in reset_terms or "randomize_dummy_rag_led_pose" in reset_terms
    )

    def maybe_randomize_led_target_anchor():
        if args_cli.fixed_led_target_anchor:
            return
        if task_randomizes_led_target_anchor:
            return
        env_ids = torch.arange(env.num_envs, dtype=torch.int64, device=env.device)
        randomize_led_target_anchor_pose(
            env,
            env_ids,
            offset_range=LED_TARGET_ANCHOR_OFFSET_RANGE,
            led_anchor_prim_expr="/World/envs/env_.*/DummyRag/LedTargetAnchor",
            rag_body_prim_expr="/World/envs/env_.*/DummyRag",
        )
        if env.sim.has_rtx_sensors():
            env.sim.render()

    def print_led_target_anchor_after_reset(reset_label: str):
        nonlocal led_target_anchor_reference_pose

        if not args_cli.debug_led_target_anchor:
            return

        pose = get_led_target_anchor_root_pose(env, [0])
        if pose is None:
            print(f"[LedTargetAnchorDebug] {reset_label}: pose could not be read.")
            return

        pose = torch.as_tensor(pose, device=env.device).reshape(-1, 7)[0]
        pos = pose[:3].detach().cpu()
        message = f"[LedTargetAnchorDebug] {reset_label}: pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})"
        if led_target_anchor_reference_pose is None:
            led_target_anchor_reference_pose = pose.detach().clone()
            message += " reference=set"
        else:
            pos_delta = torch.linalg.norm(pose[:3] - led_target_anchor_reference_pose[:3]).item()
            quat_delta = torch.linalg.norm(pose[3:7] - led_target_anchor_reference_pose[3:7]).item()
            message += f" delta_from_first_reset_pos={pos_delta:.6f}, delta_quat={quat_delta:.6f}"
        print(message)

    def reset_episode(*, reset_sim: bool, reset_recorder: bool, reset_message: str | None = None):
        nonlocal should_reset, success_step_count

        if reset_sim:
            env.sim.reset()
        if reset_recorder and recording_enabled:
            env.recorder_manager.reset()
        env.reset()
        maybe_randomize_led_target_anchor()
        print_led_target_anchor_after_reset(reset_message or "reset")
        teleop.reset()
        should_reset = False
        success_step_count = 0
        if reset_message is not None:
            print(f"{reset_message} Total successes so far: {total_success_count}.")

    print(teleop)
    print(f"Arm rotation is enabled with IK rotation scale {args_cli.arm_rotation_scale:.2f}.")
    print("Press J to print the current right-arm/head joint positions in radians.")
    if not args_cli.disable_fabric:
        print("Fabric is enabled, so Isaac Sim inspectors may not show live joint states reliably. Use --disable_fabric if needed.")
    if recording_enabled:
        if mimic_recording_enabled:
            print(f"Recording mimic-compatible successful demos to {dataset_file}")
        else:
            print(f"Recording successful demos to {dataset_file}")
        if args_cli.num_demos > 0:
            print(f"The app will stop after saving {args_cli.num_demos} successful demos.")
    else:
        print("Demo recording is disabled. Pass --dataset_file or --num_demos to save successful episodes.")
    if args_cli.fixed_led_target_anchor:
        print("LedTargetAnchor randomization is disabled; using the authored fixed LED pose on every reset.")
    elif task_randomizes_led_target_anchor:
        print("LedTargetAnchor randomization is enabled on every reset.")
    else:
        print("LedTargetAnchor randomization will be applied by this teleop script on every reset.")
    print("The task resets automatically after success. Press R to reset manually.")

    reset_episode(reset_sim=False, reset_recorder=False)

    while simulation_app.is_running():
        with torch.inference_mode():
            env.step(teleop.advance().unsqueeze(0))

            if success_term is not None and bool(success_term.func(env, **success_term.params)[0]):
                success_step_count += 1
                if success_step_count >= args_cli.num_success_steps:
                    total_success_count += 1
                    if recording_enabled:
                        env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                        env.recorder_manager.set_success_to_episodes(
                            [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                        )
                        env.recorder_manager.export_episodes([0])
                        recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                        print(
                            f"Success triggered: total successes={total_success_count}, "
                            f"saved demo #{recorded_demo_count} to {dataset_file}."
                        )
                    else:
                        print(
                            f"Success triggered: total successes={total_success_count}. "
                            "The red LED entered the camera center region."
                        )

                    reset_episode(
                        reset_sim=True,
                        reset_recorder=True,
                        reset_message="Auto reset complete.",
                    )

                    if recording_enabled and args_cli.num_demos > 0 and recorded_demo_count >= args_cli.num_demos:
                        print(f"Saved all {recorded_demo_count} requested demos. Exiting.")
                        break
                    continue
            else:
                success_step_count = 0

            if should_reset:
                print("Manual reset requested.")
                reset_episode(
                    reset_sim=True,
                    reset_recorder=True,
                    reset_message="Manual reset complete.",
                )

            if env.sim.is_stopped():
                break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
