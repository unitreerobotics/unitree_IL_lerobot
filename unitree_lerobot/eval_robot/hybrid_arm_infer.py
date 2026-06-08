#!/usr/bin/env python3
"""Hybrid ACT inference using dataset images/Dex3 state and real G1 arm state.

The policy still receives the 28-dimensional schema used during training:

    dataset camera frames + real arm state[0:14] + dataset Dex3 state[14:28]

Only the first 14 dimensions of the predicted action chunk are considered.
The default mode is read-only dry-run. No Dex3 controller is initialized.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.utils.utils import get_safe_torch_device, init_logging

from unitree_lerobot.eval_robot.hybrid_arm_utils import (
    ARM_DOF,
    FULL_STATE_DOF,
    chunk_timestep_range,
    compose_hybrid_state,
    extract_arm_chunk,
    limit_arm_target,
)
from unitree_lerobot.eval_robot.utils.utils import extract_observation


CONTROL_CONFIRMATION = "SEND_TO_REAL_G1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", required=True, help="ACT checkpoint/pretrained_model directory.")
    parser.add_argument(
        "--repo-id",
        default="unitreerobotics/G1_Dex3_ToastedBread_Dataset",
        help="LeRobot dataset supplying images and the synthetic Dex3 state.",
    )
    parser.add_argument("--root", default=None, help="Optional local LeRobot dataset root.")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0, help="Frame offset inside the selected episode.")
    parser.add_argument("--max-policy-steps", type=int, default=None)
    parser.add_argument("--actions-per-inference", type=int, default=75)
    parser.add_argument(
        "--prefetch-threshold",
        type=float,
        default=0.5,
        help="Start background inference when the action queue falls to this fraction of actions-per-inference.",
    )
    parser.add_argument(
        "--synchronous-inference",
        action="store_true",
        help="Disable background prefetch and reproduce the old blocking chunk loop.",
    )
    parser.add_argument("--frequency", type=float, default=30.0)
    parser.add_argument("--max-action-delta-rad", type=float, default=0.05)
    parser.add_argument("--state-timeout-s", type=float, default=0.25)
    parser.add_argument("--dds-wait-timeout-s", type=float, default=10.0)
    parser.add_argument("--initialization-timeout-s", type=float, default=30.0)
    parser.add_argument(
        "--initialization-speed-rad-s",
        type=float,
        default=0.1,
        help="Maximum per-joint speed while moving to the dataset initial pose.",
    )
    parser.add_argument(
        "--initialization-max-tracking-error-rad",
        type=float,
        default=0.05,
        help="Maximum gap between the initialization trajectory and measured arm state.",
    )
    parser.add_argument("--network-interface", default=None, help="DDS network interface, for example eth0.")
    parser.add_argument("--motion", action="store_true", help="Use rt/arm_sdk instead of rt/lowcmd.")
    parser.add_argument("--send-actions", action="store_true", help="Actually publish arm targets. Default is dry-run.")
    parser.add_argument(
        "--initialize-from-dataset",
        action="store_true",
        help="Before inference, slowly command the dataset arm pose at start-frame.",
    )
    parser.add_argument(
        "--control-confirmation",
        default="",
        help=f"Required with --send-actions; must equal {CONTROL_CONFIRMATION!r}.",
    )
    parser.add_argument("--output-root", default="hybrid_arm_results")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.episode < 0 or args.start_frame < 0:
        raise ValueError("episode and start-frame must be non-negative")
    if args.actions_per_inference <= 0:
        raise ValueError("actions-per-inference must be greater than zero")
    if not 0 < args.prefetch_threshold < 1:
        raise ValueError("prefetch-threshold must be between zero and one")
    if args.frequency <= 0:
        raise ValueError("frequency must be greater than zero")
    if args.state_timeout_s <= 0 or args.dds_wait_timeout_s <= 0 or args.initialization_timeout_s <= 0:
        raise ValueError("All timeout values must be greater than zero")
    if args.initialization_speed_rad_s <= 0 or args.initialization_max_tracking_error_rad <= 0:
        raise ValueError("Initialization speed and tracking error must be greater than zero")
    if args.max_policy_steps is not None and args.max_policy_steps <= 0:
        raise ValueError("max-policy-steps must be greater than zero")
    if args.send_actions and args.control_confirmation != CONTROL_CONFIRMATION:
        raise ValueError(
            f"Real control requires --control-confirmation={CONTROL_CONFIRMATION}. "
            "Run without --send-actions for read-only dry-run."
        )
    if args.initialize_from_dataset and not args.send_actions:
        raise ValueError("--initialize-from-dataset is only valid with --send-actions")


def make_run_dir(output_root: Path, run_name: str | None, episode: int) -> Path:
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{timestamp}_episode-{episode:03d}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def episode_bounds(dataset: LeRobotDataset, episode: int, start_frame: int) -> tuple[int, int]:
    if episode >= dataset.meta.total_episodes:
        raise ValueError(f"Episode {episode} is out of range [0, {dataset.meta.total_episodes - 1}]")
    episode_from = int(dataset.meta.episodes["dataset_from_index"][episode])
    episode_to = int(dataset.meta.episodes["dataset_to_index"][episode])
    start = episode_from + start_frame
    if start >= episode_to:
        raise ValueError(f"start-frame {start_frame} is outside episode {episode}")
    return start, episode_to


def load_policy_and_processors(policy_path: Path, dataset: LeRobotDataset):
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    if policy_cfg.type != "act":
        raise ValueError(f"This entry point requires an ACT checkpoint, got policy type {policy_cfg.type!r}")
    state_feature = policy_cfg.input_features.get("observation.state")
    action_feature = policy_cfg.output_features.get("action")
    if state_feature is None or tuple(state_feature.shape) != (FULL_STATE_DOF,):
        raise ValueError(f"Checkpoint observation.state must have shape ({FULL_STATE_DOF},)")
    if action_feature is None or tuple(action_feature.shape) != (FULL_STATE_DOF,):
        raise ValueError(f"Checkpoint action must have shape ({FULL_STATE_DOF},)")
    policy_cfg.pretrained_path = policy_path
    device = get_safe_torch_device(policy_cfg.device, log=True)
    policy = make_policy(cfg=policy_cfg, ds_meta=dataset.meta)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_path,
        dataset_stats=rename_stats(dataset.meta.stats, {}),
        preprocessor_overrides={
            "device_processor": {"device": policy_cfg.device},
            "rename_observations_processor": {"rename_map": {}},
        },
    )
    return policy_cfg, policy, preprocessor, postprocessor, device


def predict_full_action_chunk(
    observation: dict[str, Any],
    task: str,
    policy,
    preprocessor,
    postprocessor,
    device: torch.device,
) -> np.ndarray:
    """Run ACT once and return the unnormalized chunk with shape (T, 28)."""
    batch = {}
    for name, value in observation.items():
        if not hasattr(value, "unsqueeze"):
            continue
        batch[name] = value.unsqueeze(0).to(device)
    batch["task"] = task or ""
    batch["robot_type"] = ""

    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type)
        if device.type == "cuda" and policy.config.use_amp
        else nullcontext(),
    ):
        processed = preprocessor(batch)
        chunk = policy.predict_action_chunk(processed)
        chunk = postprocessor(chunk)

    chunk_np = chunk.squeeze(0).detach().cpu().numpy()
    if chunk_np.ndim != 2 or chunk_np.shape[1] != FULL_STATE_DOF:
        raise ValueError(f"Expected policy output (T, {FULL_STATE_DOF}), got {chunk_np.shape}")
    return chunk_np


@dataclass
class QueuedArmAction:
    predicted: np.ndarray
    chunk_offset: int
    inference_anchor: int
    inference_s: float
    report_inference: bool = False


@dataclass
class InferenceResult:
    anchor: int
    arm_chunk: np.ndarray
    inference_s: float


def make_hybrid_observation(dataset, dataset_index: int, real_arm: np.ndarray) -> tuple[dict[str, Any], str]:
    step = dataset[dataset_index]
    dataset_state = step["observation.state"].detach().cpu().numpy()
    hybrid_state = compose_hybrid_state(dataset_state, real_arm)
    observation = extract_observation(step)
    observation = {
        name: value.clone() if isinstance(value, torch.Tensor) else value for name, value in observation.items()
    }
    observation["observation.state"] = torch.as_tensor(
        hybrid_state,
        dtype=step["observation.state"].dtype,
    )
    return observation, step["task"]


def infer_arm_chunk(
    anchor: int,
    observation: dict[str, Any],
    task: str,
    policy,
    preprocessor,
    postprocessor,
    device: torch.device,
) -> InferenceResult:
    preprocessor.reset()
    postprocessor.reset()
    inference_start = time.perf_counter()
    full_chunk = predict_full_action_chunk(
        observation,
        task,
        policy,
        preprocessor,
        postprocessor,
        device,
    )
    return InferenceResult(
        anchor=anchor,
        arm_chunk=extract_arm_chunk(full_chunk),
        inference_s=time.perf_counter() - inference_start,
    )


def merge_inference_result(
    action_queue: dict[int, QueuedArmAction],
    result: InferenceResult,
    current_timestep: int,
    actions_per_inference: int,
    stop_timestep: int,
) -> int:
    timesteps = chunk_timestep_range(
        result.anchor,
        len(result.arm_chunk),
        actions_per_inference,
        current_timestep,
    )
    inserted_timesteps = []
    for timestep in timesteps:
        if timestep >= stop_timestep:
            break
        offset = timestep - result.anchor
        action_queue[timestep] = QueuedArmAction(
            predicted=result.arm_chunk[offset],
            chunk_offset=offset,
            inference_anchor=result.anchor,
            inference_s=result.inference_s,
        )
        inserted_timesteps.append(timestep)
    if inserted_timesteps:
        action_queue[min(inserted_timesteps)].report_inference = True
    return len(inserted_timesteps)


def setup_arm_io(args: argparse.Namespace):
    if not args.send_actions:
        from unitree_lerobot.eval_robot.robot_control.arm_state_reader import G1ArmStateReader

        reader = G1ArmStateReader(
            network_interface=args.network_interface,
            timeout_s=args.dds_wait_timeout_s,
        )
        return reader, None, None

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    ChannelFactoryInitialize(0, args.network_interface)
    report_motion_mode()
    if args.motion:
        print(
            "G1 arm SDK preflight: the robot must be in Regular motion-control mode "
            "(latest R3 sequence: L2+B, L2+UP, then R1+X), not Running or Debug mode."
        )
    from unitree_lerobot.eval_robot.robot_control.robot_arm import G1_29_ArmController
    from unitree_lerobot.eval_robot.robot_control.robot_arm_ik import G1_29_ArmIK

    controller = G1_29_ArmController(
        motion_mode=args.motion,
        simulation_mode=False,
        initialize_dds=False,
    )
    arm_ik = G1_29_ArmIK()
    current = controller.get_current_dual_arm_q()
    controller.ctrl_dual_arm(current, arm_ik.solve_tau(current))
    return controller, controller, arm_ik


def report_motion_mode() -> None:
    """Print the current MotionSwitcher state without changing robot mode."""
    try:
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

        client = MotionSwitcherClient()
        client.SetTimeout(3.0)
        client.Init()
        code, mode = client.CheckMode()
        print(f"G1 MotionSwitcher: code={code}, mode={mode}")
    except Exception as exc:
        print(f"WARNING: Could not query G1 MotionSwitcher mode: {exc}")


def write_config(run_dir: Path, args: argparse.Namespace, policy_cfg, start: int, stop: int) -> None:
    config = vars(args).copy()
    config.update(
        {
            "policy_path": str(Path(args.policy_path).resolve()),
            "dataset_start_index": start,
            "dataset_stop_index": stop,
            "policy_type": policy_cfg.type,
            "chunk_size": getattr(policy_cfg, "chunk_size", None),
            "dry_run": not args.send_actions,
        }
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def read_arm_state(state_source, max_age_s: float) -> np.ndarray:
    """Read from either the read-only subscriber or the existing arm controller."""
    try:
        return state_source.get_current_dual_arm_q(max_age_s=max_age_s)
    except TypeError:
        state = np.asarray(state_source.get_current_dual_arm_q(), dtype=np.float32)
        if state.shape != (ARM_DOF,) or not np.all(np.isfinite(state)):
            raise RuntimeError(f"Invalid G1 arm state: shape={state.shape}")
        return state


def initialize_arm(
    controller,
    arm_ik,
    target: np.ndarray,
    frequency: float,
    speed_rad_s: float,
    timeout_s: float,
    max_tracking_error_rad: float,
    motion_mode: bool,
) -> None:
    start_s = time.monotonic()
    deadline = start_s + timeout_s
    max_delta_rad = speed_rad_s / frequency
    initial_state = read_arm_state(controller, max_age_s=0.25)
    trajectory_target = initial_state.copy()
    last_progress_s = time.monotonic()
    while True:
        now = time.monotonic()
        if now >= deadline:
            raise TimeoutError(f"Robot did not reach the dataset initial pose within {timeout_s:.1f}s")
        current = read_arm_state(controller, max_age_s=0.25)
        trajectory_target = limit_arm_target(target, trajectory_target, max_delta_rad)
        command = limit_arm_target(trajectory_target, current, max_tracking_error_rad)
        tau = arm_ik.solve_tau(command)
        controller.ctrl_dual_arm(command, tau)
        max_error = float(np.max(np.abs(target - current)))
        if max_error < 0.02:
            return
        if now - last_progress_s >= 1.0:
            max_motion = float(np.max(np.abs(current - initial_state)))
            print(f"Initializing arms: max_error={max_error:.3f} rad, moved={max_motion:.3f} rad")
            last_progress_s = now
            if now + 1.0 / frequency >= deadline:
                continue
            if now - start_s >= 3.0 and max_motion < 0.005:
                if motion_mode:
                    raise RuntimeError(
                        "Commands were published to rt/arm_sdk but no arm motion was detected. "
                        "Confirm the G1 is in Regular motion-control mode (R1+X), not Running mode (R2+A) "
                        "or Debug mode (L2+R2). MotionSwitcher name 'ai' alone does not identify this sub-mode. "
                        "No robot mode was changed automatically."
                    )
                raise RuntimeError(
                    "Commands were published to rt/lowcmd but no arm motion was detected. "
                    "If a G1 motion service is active, rerun with --motion to use rt/arm_sdk."
                )
        time.sleep(1.0 / frequency)


def run(args: argparse.Namespace) -> Path:
    validate_args(args)
    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root)
    start, stop = episode_bounds(dataset, args.episode, args.start_frame)
    policy_cfg, policy, preprocessor, postprocessor, device = load_policy_and_processors(
        Path(args.policy_path), dataset
    )
    if getattr(policy_cfg, "chunk_size", 0) < args.actions_per_inference:
        raise ValueError("actions-per-inference cannot exceed the ACT chunk size")

    run_dir = make_run_dir(Path(args.output_root), args.run_name, args.episode)
    write_config(run_dir, args, policy_cfg, start, stop)

    state_source = controller = arm_ik = None
    csv_path = run_dir / "steps.csv"
    fieldnames = [
        "policy_step",
        "dataset_index",
        "dataset_frame",
        "chunk_offset",
        "inference_anchor",
        "inference_s",
        "action_queue_size",
        "queue_wait_s",
        "loop_s",
        "max_abs_prediction_delta",
        "max_abs_command_delta",
        "sent_to_robot",
        "real_arm_state",
        "dataset_arm_state",
        "dataset_dex3_state",
        "predicted_arm_action",
        "limited_arm_target",
        "dataset_arm_action",
    ]

    try:
        state_source, controller, arm_ik = setup_arm_io(args)
        first_step = dataset[start]
        first_state = first_step["observation.state"].detach().cpu().numpy()
        if first_state.shape != (FULL_STATE_DOF,):
            raise ValueError(f"Dataset state must have shape ({FULL_STATE_DOF},), got {first_state.shape}")

        if args.initialize_from_dataset:
            initialize_arm(
                controller,
                arm_ik,
                first_state[:ARM_DOF],
                frequency=args.frequency,
                speed_rad_s=args.initialization_speed_rad_s,
                timeout_s=args.initialization_timeout_s,
                max_tracking_error_rad=args.initialization_max_tracking_error_rad,
                motion_mode=args.motion,
            )
            print("Dataset initial arm pose reached. Policy has not started yet.")
            confirmation = input("Enter 's' to start policy control, or anything else to stop while holding pose: ")
            if confirmation.strip().lower() != "s":
                print("Policy control cancelled.")
                return run_dir

        dataset_index = start
        policy_step = 0
        execution_stop = stop
        if args.max_policy_steps is not None:
            execution_stop = min(execution_stop, start + args.max_policy_steps)
        prefetch_count = max(1, int(np.ceil(args.actions_per_inference * args.prefetch_threshold)))
        action_queue: dict[int, QueuedArmAction] = {}
        inference_future: Future[InferenceResult] | None = None

        with csv_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="act-inference") as executor:

                def submit_inference(anchor: int) -> Future[InferenceResult]:
                    real_arm = read_arm_state(state_source, max_age_s=args.state_timeout_s)
                    observation, task = make_hybrid_observation(dataset, anchor, real_arm)
                    return executor.submit(
                        infer_arm_chunk,
                        anchor,
                        observation,
                        task,
                        policy,
                        preprocessor,
                        postprocessor,
                        device,
                    )

                inference_future = submit_inference(dataset_index)
                print("Preparing initial ACT action chunk...")

                while dataset_index < execution_stop:
                    loop_start = time.perf_counter()
                    queue_wait_s = 0.0

                    if inference_future is not None and inference_future.done():
                        result = inference_future.result()
                        inserted = merge_inference_result(
                            action_queue,
                            result,
                            dataset_index,
                            args.actions_per_inference,
                            execution_stop,
                        )
                        print(
                            f"inference anchor={result.anchor - start} time={result.inference_s * 1000:.1f}ms "
                            f"usable_actions={inserted} queue={len(action_queue)}"
                        )
                        inference_future = None

                    for stale_timestep in [t for t in action_queue if t < dataset_index]:
                        del action_queue[stale_timestep]

                    if dataset_index not in action_queue:
                        if inference_future is None:
                            inference_future = submit_inference(dataset_index)
                        wait_start = time.perf_counter()
                        result = inference_future.result()
                        queue_wait_s = time.perf_counter() - wait_start
                        inserted = merge_inference_result(
                            action_queue,
                            result,
                            dataset_index,
                            args.actions_per_inference,
                            execution_stop,
                        )
                        print(
                            f"inference anchor={result.anchor - start} time={result.inference_s * 1000:.1f}ms "
                            f"usable_actions={inserted} queue={len(action_queue)} "
                            f"wait={queue_wait_s * 1000:.1f}ms"
                        )
                        inference_future = None
                        if dataset_index not in action_queue:
                            raise RuntimeError("Inference completed without an action for the current timestep")

                    queued_action = action_queue.pop(dataset_index)
                    action_step = dataset[dataset_index]
                    action_dataset_state = action_step["observation.state"].detach().cpu().numpy()
                    current_arm = read_arm_state(state_source, max_age_s=args.state_timeout_s)
                    predicted = queued_action.predicted
                    limited = limit_arm_target(predicted, current_arm, args.max_action_delta_rad)

                    if args.send_actions:
                        tau = arm_ik.solve_tau(limited)
                        controller.ctrl_dual_arm(limited, tau)

                    dataset_action = action_step["action"].detach().cpu().numpy()
                    writer.writerow(
                        {
                            "policy_step": policy_step,
                            "dataset_index": dataset_index,
                            "dataset_frame": int(action_step["frame_index"].item()),
                            "chunk_offset": queued_action.chunk_offset,
                            "inference_anchor": queued_action.inference_anchor - start,
                            "inference_s": queued_action.inference_s if queued_action.report_inference else 0.0,
                            "action_queue_size": len(action_queue),
                            "queue_wait_s": queue_wait_s,
                            "loop_s": time.perf_counter() - loop_start,
                            "max_abs_prediction_delta": float(np.max(np.abs(predicted - current_arm))),
                            "max_abs_command_delta": float(np.max(np.abs(limited - current_arm))),
                            "sent_to_robot": args.send_actions,
                            "real_arm_state": json.dumps(current_arm.tolist()),
                            "dataset_arm_state": json.dumps(action_dataset_state[:ARM_DOF].tolist()),
                            "dataset_dex3_state": json.dumps(action_dataset_state[ARM_DOF:].tolist()),
                            "predicted_arm_action": json.dumps(predicted.tolist()),
                            "limited_arm_target": json.dumps(limited.tolist()),
                            "dataset_arm_action": json.dumps(dataset_action[:ARM_DOF].tolist()),
                        }
                    )
                    csv_file.flush()

                    policy_step += 1
                    dataset_index += 1

                    remaining_actions = sum(t >= dataset_index for t in action_queue)
                    should_prefetch = (
                        not args.synchronous_inference and remaining_actions <= prefetch_count
                    ) or (args.synchronous_inference and remaining_actions == 0)
                    if (
                        should_prefetch
                        and inference_future is None
                        and dataset_index < execution_stop
                    ):
                        inference_future = submit_inference(dataset_index)

                    if policy_step % 30 == 0 or dataset_index == execution_stop:
                        print(
                            f"step={policy_step} dataset_index={dataset_index - start} "
                            f"queue={remaining_actions} prefetch={inference_future is not None} "
                            f"dry_run={not args.send_actions}"
                        )

                    elapsed = time.perf_counter() - loop_start
                    time.sleep(max(0.0, 1.0 / args.frequency - elapsed))
    finally:
        if args.send_actions and controller is not None and arm_ik is not None:
            try:
                hold_state = read_arm_state(controller, max_age_s=args.state_timeout_s)
                controller.ctrl_dual_arm(hold_state, arm_ik.solve_tau(hold_state))
                time.sleep(0.1)
            except Exception as exc:
                print(f"Warning: failed to command final hold pose: {exc}")
        if not args.send_actions and state_source is not None:
            state_source.close()

    return run_dir


def main() -> None:
    init_logging()
    args = parse_args()
    run_dir = run(args)
    print(f"Hybrid arm inference results: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
