"""Offline evaluation on a recorded dataset; optionally replays actions on the real robot."""

import torch
import tqdm
import logging
import time
import numpy as np
import matplotlib.pyplot as plt

from pprint import pformat
from typing import Any
from dataclasses import asdict
from torch import nn
from contextlib import nullcontext

from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.utils import get_safe_torch_device, init_logging
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor.rename_processor import rename_stats
from lerobot.processor import PolicyAction, PolicyProcessorPipeline

from unitree_lerobot.eval_robot.robot import UnitreeRobot
from unitree_lerobot.eval_robot.utils.utils import (
    EvalRealConfig,
    extract_observation,
    predict_action,
)
from unitree_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger, visualization_data

import logging_mp

logger_mp = logging_mp.getLogger(__name__)
logger_mp.setLevel(logging_mp.INFO)


def eval_policy(
    cfg: EvalRealConfig,
    dataset: LeRobotDataset,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
):
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."
    logger_mp.info(f"Arguments: {cfg}")

    rerun_logger = RerunLogger() if cfg.visualization else None

    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    from_idx = dataset.meta.episodes["dataset_from_index"][0]
    to_idx = dataset.meta.episodes["dataset_to_index"][0]
    step = dataset[from_idx]

    ground_truth_actions: list[np.ndarray] = []
    predicted_actions: list[np.ndarray] = []

    robot: UnitreeRobot | None = None
    if cfg.send_real_robot:
        robot = UnitreeRobot(cfg)
        init_arm_pose = step["observation.state"][: robot.arm_dof].cpu().numpy()

    user_input = input("Please enter the start signal (enter 's' to start the subsequent program):")
    if user_input.lower() != "s":
        return

    try:
        if robot is not None:
            logger_mp.info("Initializing robot to starting pose...")
            robot.send_arm(init_arm_pose)
            time.sleep(1)

        for step_idx in tqdm.tqdm(range(from_idx, to_idx)):
            loop_start_time = time.perf_counter()

            step = dataset[step_idx]
            observation = extract_observation(step)

            action = predict_action(
                observation,
                policy,
                get_safe_torch_device(policy.config.device),
                preprocessor,
                postprocessor,
                policy.config.use_amp,
                step["task"],
                use_dataset=True,
                robot_type=None,
            )
            action_np = action.cpu().numpy()

            ground_truth_actions.append(step["action"].numpy())
            predicted_actions.append(action_np)

            if robot is not None:
                robot.execute_action(action_np)

            if rerun_logger is not None:
                visualization_data(step_idx, observation, observation["observation.state"], action_np, rerun_logger)

            time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start_time)))
    finally:
        if robot is not None:
            robot.close()

    gt = np.array(ground_truth_actions)
    pred = np.array(predicted_actions)
    if gt.size == 0:
        return
    n_timesteps, n_dims = gt.shape

    fig, axes = plt.subplots(n_dims, 1, figsize=(12, 4 * n_dims), sharex=True)
    fig.suptitle("Ground Truth vs Predicted Actions")
    for i in range(n_dims):
        ax = axes[i] if n_dims > 1 else axes
        ax.plot(gt[:, i], label="Ground Truth", color="blue")
        ax.plot(pred[:, i], label="Predicted", color="red", linestyle="--")
        ax.set_ylabel(f"Dim {i + 1}")
        ax.legend()
    (axes[-1] if n_dims > 1 else axes).set_xlabel("Timestep")
    plt.tight_layout()
    time.sleep(1)
    plt.savefig("figure.png")


@parser.wrap()
def eval_main(cfg: EvalRealConfig):
    logging.info(pformat(asdict(cfg)))

    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Making policy.")

    dataset = LeRobotDataset(repo_id=cfg.repo_id)

    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=rename_stats(dataset.meta.stats, cfg.rename_map),
        preprocessor_overrides={
            "device_processor": {"device": cfg.policy.device},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        eval_policy(cfg, dataset, policy, preprocessor, postprocessor)

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
