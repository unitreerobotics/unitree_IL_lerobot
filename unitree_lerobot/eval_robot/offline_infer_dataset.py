#!/usr/bin/env python
"""Offline inference report for a LeRobot dataset and a trained policy.

This script never sends actions to a robot. It only loads recorded dataset
observations, runs the policy, compares predicted actions to recorded actions,
and saves plots plus metrics in a run directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.utils.utils import get_safe_torch_device, init_logging

from unitree_lerobot.eval_robot.utils.utils import extract_observation, predict_action


ACTION_GROUPS = {
    "left_arm": slice(0, 7),
    "right_arm": slice(7, 14),
    "left_dex3": slice(14, 21),
    "right_dex3": slice(21, 28),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline dataset inference and save a report.")
    parser.add_argument(
        "--repo-id",
        default="unitreerobotics/G1_Dex3_ToastedBread_Dataset",
        help="LeRobot dataset repo id.",
    )
    parser.add_argument(
        "--policy-path",
        required=True,
        help="Path to checkpoint/pretrained_model containing model.safetensors and processors.",
    )
    parser.add_argument("--episode", type=int, default=0, help="Episode index to evaluate.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame cap for quick smoke tests. Defaults to the full episode.",
    )
    parser.add_argument(
        "--output-root",
        default="offline_infer_results",
        help="Directory where report run folders are created.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run folder name. Defaults to a timestamped name.",
    )
    parser.add_argument(
        "--mode",
        choices=("queued", "reset_each_frame"),
        default="queued",
        help=(
            "queued matches deployment-style ACT action queue. reset_each_frame clears the queue for every "
            "frame and compares only the first predicted action."
        ),
    )
    return parser.parse_args()


def make_run_dir(output_root: Path, run_name: str | None, episode: int, mode: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name is None:
        run_name = f"{timestamp}_episode-{episode:03d}_{mode}"
    run_dir = output_root / run_name
    (run_dir / "plots").mkdir(parents=True, exist_ok=False)
    return run_dir


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, slice):
        return {"start": value.start, "stop": value.stop}
    return value


def load_policy_and_processors(policy_path: Path, dataset: LeRobotDataset):
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    device = get_safe_torch_device(policy_cfg.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

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


def episode_bounds(dataset: LeRobotDataset, episode: int, max_frames: int | None) -> tuple[int, int]:
    if episode < 0 or episode >= dataset.meta.total_episodes:
        raise ValueError(f"Episode {episode} is out of range [0, {dataset.meta.total_episodes - 1}]")
    from_idx = int(dataset.meta.episodes["dataset_from_index"][episode])
    to_idx = int(dataset.meta.episodes["dataset_to_index"][episode])
    if max_frames is not None:
        to_idx = min(to_idx, from_idx + max_frames)
    return from_idx, to_idx


def run_inference(
    dataset: LeRobotDataset,
    policy,
    preprocessor,
    postprocessor,
    device: torch.device,
    episode: int,
    max_frames: int | None,
    mode: str,
) -> dict[str, Any]:
    from_idx, to_idx = episode_bounds(dataset, episode, max_frames)

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    ground_truth_actions = []
    predicted_actions = []
    frame_indices = []
    timestamps = []

    start_s = time.perf_counter()
    iterator = tqdm.tqdm(range(from_idx, to_idx), desc=f"episode {episode}")
    amp_ctx = torch.autocast(device_type=device.type) if device.type == "cuda" and policy.config.use_amp else nullcontext()

    with torch.no_grad(), amp_ctx:
        for step_idx in iterator:
            if mode == "reset_each_frame":
                policy.reset()
                preprocessor.reset()
                postprocessor.reset()

            step = dataset[step_idx]
            observation = extract_observation(step)
            action = predict_action(
                observation,
                policy,
                device,
                preprocessor,
                postprocessor,
                policy.config.use_amp,
                step["task"],
                use_dataset=True,
                robot_type=None,
            )

            ground_truth_actions.append(step["action"].detach().cpu().numpy())
            predicted_actions.append(action.detach().cpu().numpy())
            frame_indices.append(int(step["frame_index"].item()))
            timestamps.append(float(step["timestamp"].item()))

    return {
        "ground_truth": np.asarray(ground_truth_actions, dtype=np.float32),
        "predicted": np.asarray(predicted_actions, dtype=np.float32),
        "frame_indices": np.asarray(frame_indices, dtype=np.int64),
        "timestamps": np.asarray(timestamps, dtype=np.float32),
        "from_idx": from_idx,
        "to_idx": to_idx,
        "elapsed_s": time.perf_counter() - start_s,
    }


def compute_metrics(gt: np.ndarray, pred: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    err = pred - gt
    abs_err = np.abs(err)
    squared_err = err * err

    dim_rows = []
    for dim in range(gt.shape[1]):
        dim_rows.append(
            {
                "dim": dim,
                "dim_1based": dim + 1,
                "mae": float(abs_err[:, dim].mean()),
                "rmse": float(np.sqrt(squared_err[:, dim].mean())),
                "max_abs_error": float(abs_err[:, dim].max()),
                "gt_mean": float(gt[:, dim].mean()),
                "pred_mean": float(pred[:, dim].mean()),
            }
        )

    group_metrics = {}
    for name, group_slice in ACTION_GROUPS.items():
        group_abs = abs_err[:, group_slice]
        group_sq = squared_err[:, group_slice]
        group_metrics[name] = {
            "dims": [group_slice.start, group_slice.stop - 1],
            "mae": float(group_abs.mean()),
            "rmse": float(np.sqrt(group_sq.mean())),
            "max_abs_error": float(group_abs.max()),
        }

    summary = {
        "num_frames": int(gt.shape[0]),
        "action_dim": int(gt.shape[1]),
        "overall": {
            "mae": float(abs_err.mean()),
            "rmse": float(np.sqrt(squared_err.mean())),
            "max_abs_error": float(abs_err.max()),
        },
        "groups": group_metrics,
    }
    return summary, dim_rows


def write_metrics(run_dir: Path, summary: dict[str, Any], dim_rows: list[dict[str, Any]]) -> None:
    with (run_dir / "metrics_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=to_jsonable)

    with (run_dir / "metrics_by_dim.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dim_rows[0].keys()))
        writer.writeheader()
        writer.writerows(dim_rows)


def save_arrays(run_dir: Path, result: dict[str, Any]) -> None:
    np.savez_compressed(
        run_dir / "predictions.npz",
        ground_truth=result["ground_truth"],
        predicted=result["predicted"],
        error=result["predicted"] - result["ground_truth"],
        frame_indices=result["frame_indices"],
        timestamps=result["timestamps"],
    )


def plot_all_dims(run_dir: Path, gt: np.ndarray, pred: np.ndarray) -> None:
    n_dims = gt.shape[1]
    fig, axes = plt.subplots(n_dims, 1, figsize=(12, max(3 * n_dims, 12)), sharex=True)
    fig.suptitle("Ground Truth vs Predicted Actions")
    x = np.arange(gt.shape[0])

    for dim in range(n_dims):
        ax = axes[dim] if n_dims > 1 else axes
        ax.plot(x, gt[:, dim], label="Ground Truth", color="blue", linewidth=1.2)
        ax.plot(x, pred[:, dim], label="Predicted", color="red", linestyle="--", linewidth=1.2)
        ax.set_ylabel(f"Dim {dim + 1}")
        ax.legend(loc="upper right", fontsize=7)

    axes[-1].set_xlabel("Frame in evaluated segment")
    plt.tight_layout()
    fig.savefig(run_dir / "plots" / "actions_all_dims.png", dpi=150)
    plt.close(fig)


def plot_group_overview(run_dir: Path, gt: np.ndarray, pred: np.ndarray) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=True)
    fig.suptitle("Action Groups: Ground Truth vs Predicted")
    x = np.arange(gt.shape[0])

    for ax, (group_name, group_slice) in zip(axes, ACTION_GROUPS.items(), strict=True):
        for dim in range(group_slice.start, group_slice.stop):
            ax.plot(x, gt[:, dim], color="blue", alpha=0.35, linewidth=1.0)
            ax.plot(x, pred[:, dim], color="red", alpha=0.35, linewidth=1.0, linestyle="--")
        ax.set_title(f"{group_name} dims {group_slice.start + 1}-{group_slice.stop}")
        ax.set_ylabel("Action")
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Frame in evaluated segment")
    handles = [
        plt.Line2D([0], [0], color="blue", label="Ground Truth"),
        plt.Line2D([0], [0], color="red", linestyle="--", label="Predicted"),
    ]
    fig.legend(handles=handles, loc="upper right")
    plt.tight_layout()
    fig.savefig(run_dir / "plots" / "actions_by_group.png", dpi=150)
    plt.close(fig)


def plot_error_summary(run_dir: Path, dim_rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    dims = [row["dim_1based"] for row in dim_rows]
    maes = [row["mae"] for row in dim_rows]
    group_names = list(summary["groups"].keys())
    group_maes = [summary["groups"][name]["mae"] for name in group_names]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].bar(dims, maes, color="#4c78a8")
    axes[0].set_title("MAE by Action Dimension")
    axes[0].set_xlabel("Action dimension")
    axes[0].set_ylabel("MAE")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(group_names, group_maes, color="#f58518")
    axes[1].set_title("MAE by Action Group")
    axes[1].set_ylabel("MAE")
    axes[1].grid(axis="y", alpha=0.25)

    plt.tight_layout()
    fig.savefig(run_dir / "plots" / "mae_summary.png", dpi=150)
    plt.close(fig)


def write_readme(run_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Offline Inference Report",
        "",
        "Artifacts:",
        "",
        "- `config.json`: run configuration.",
        "- `metrics_summary.json`: overall and group metrics.",
        "- `metrics_by_dim.csv`: per-action-dimension metrics.",
        "- `predictions.npz`: ground truth, prediction, error, frame indices and timestamps.",
        "- `plots/actions_all_dims.png`: all 28 action dimensions.",
        "- `plots/actions_by_group.png`: grouped arm/hand overview.",
        "- `plots/mae_summary.png`: MAE bars.",
        "",
        "Quick summary:",
        "",
        f"- Frames: `{summary['num_frames']}`",
        f"- Overall MAE: `{summary['overall']['mae']:.6f}`",
        f"- Overall RMSE: `{summary['overall']['rmse']:.6f}`",
    ]
    for group_name, metrics in summary["groups"].items():
        lines.append(f"- {group_name} MAE: `{metrics['mae']:.6f}`")

    (run_dir / "README.md").write_text("\n".join(lines) + "\n")


def update_latest_symlink(output_root: Path, run_dir: Path) -> None:
    latest = output_root / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(run_dir.resolve())
    except OSError:
        pass


def main() -> None:
    init_logging()
    args = parse_args()

    output_root = Path(args.output_root)
    policy_path = Path(args.policy_path)
    run_dir = make_run_dir(output_root, args.run_name, args.episode, args.mode)

    dataset = LeRobotDataset(repo_id=args.repo_id)
    policy_cfg, policy, preprocessor, postprocessor, device = load_policy_and_processors(policy_path, dataset)

    config = {
        "repo_id": args.repo_id,
        "policy_path": str(policy_path.resolve()),
        "episode": args.episode,
        "max_frames": args.max_frames,
        "mode": args.mode,
        "output_root": str(output_root.resolve()),
        "run_dir": str(run_dir.resolve()),
        "policy_type": policy_cfg.type,
        "device": policy_cfg.device,
        "chunk_size": getattr(policy_cfg, "chunk_size", None),
        "n_action_steps": getattr(policy_cfg, "n_action_steps", None),
    }
    with (run_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2, default=to_jsonable)

    result = run_inference(
        dataset=dataset,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        device=device,
        episode=args.episode,
        max_frames=args.max_frames,
        mode=args.mode,
    )

    summary, dim_rows = compute_metrics(result["ground_truth"], result["predicted"])
    summary.update(
        {
            "episode": args.episode,
            "from_idx": result["from_idx"],
            "to_idx": result["to_idx"],
            "elapsed_s": result["elapsed_s"],
            "mode": args.mode,
        }
    )

    write_metrics(run_dir, summary, dim_rows)
    save_arrays(run_dir, result)
    plot_all_dims(run_dir, result["ground_truth"], result["predicted"])
    plot_group_overview(run_dir, result["ground_truth"], result["predicted"])
    plot_error_summary(run_dir, dim_rows, summary)
    write_readme(run_dir, summary)
    update_latest_symlink(output_root, run_dir)

    print(f"Saved offline inference report to: {run_dir.resolve()}")
    print(f"Overall MAE: {summary['overall']['mae']:.6f}")
    for group_name, metrics in summary["groups"].items():
        print(f"{group_name} MAE: {metrics['mae']:.6f}")


if __name__ == "__main__":
    main()
