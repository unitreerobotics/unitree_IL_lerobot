"""Simulation evaluation loop using the unified UnitreeRobot + sim-only DDS hooks."""

import time
import torch
import logging
import numpy as np

from pprint import pformat
from dataclasses import asdict
from typing import Any
from torch import nn
from contextlib import nullcontext

from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.utils import get_safe_torch_device, init_logging
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor.rename_processor import rename_stats
from lerobot.processor import PolicyAction, PolicyProcessorPipeline

from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

from unitree_lerobot.eval_robot.robot import UnitreeRobot
from unitree_lerobot.eval_robot.utils.utils import predict_action
from unitree_lerobot.eval_robot.utils.sim_savedata_utils import (
    EvalRealConfig,
    process_data_add,
    is_success,
)
from unitree_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger, visualization_data
from unitree_lerobot.eval_robot.utils.episode_writer import EpisodeWriter
from unitree_lerobot.eval_robot.utils.sim_state_topic import (
    start_sim_state_subscribe,
    start_sim_reward_subscribe,
)

import logging_mp

logger_mp = logging_mp.getLogger(__name__)
logger_mp.setLevel(logging_mp.INFO)


def _setup_sim_extras(cfg: EvalRealConfig):
    """DDS subscribers/publishers only used in the sim flow."""
    reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
    reset_pose_publisher.Init()
    sim_state_subscriber = start_sim_state_subscribe()
    sim_reward_subscriber = start_sim_reward_subscribe()
    episode_writer = None
    if getattr(cfg, "save_data", False) and getattr(cfg, "task_dir", None):
        episode_writer = EpisodeWriter(cfg.task_dir, frequency=30, image_size=[640, 480])
    return {
        "sim_state_subscriber": sim_state_subscriber,
        "sim_reward_subscriber": sim_reward_subscriber,
        "episode_writer": episode_writer,
        "reset_pose_publisher": reset_pose_publisher,
    }


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

    # Force sim flag on the cfg so UnitreeRobot initializes its subsystems in sim mode.
    if not getattr(cfg, "sim", False):
        setattr(cfg, "sim", True)

    robot = UnitreeRobot(cfg)
    sim = _setup_sim_extras(cfg)

    try:
        from_idx = dataset.meta.episodes["dataset_from_index"][0]
        step = dataset[from_idx]
        init_arm_pose = step["observation.state"][: robot.arm_dof].cpu().numpy()

        user_input = input("Enter 's' to initialize the robot and start the evaluation: ")
        if user_input.lower() != "s":
            return

        logger_mp.info("Initializing robot to starting pose...")
        robot.send_arm(init_arm_pose)
        time.sleep(1.0)

        reward_stats = {"reward_sum": 0.0, "episode_num": 0.0}
        # Back-compat dict expected by `is_success`.
        compat_iface = {"arm_ik": robot.arm_ik, "arm_ctrl": robot.arm_ctrl}

        logger_mp.info(f"Starting evaluation loop at {cfg.frequency} Hz.")
        idx = 0
        while True:
            if cfg.save_data and reward_stats["episode_num"] == 0:
                sim["episode_writer"].create_episode()

            loop_start_time = time.perf_counter()

            observation = robot.get_observation()
            state_tensor = observation["observation.state"]
            # `process_data_add` expects the raw state slices used during recording.
            left_ee, right_ee = robot.get_ee_state()
            full_ee_state = np.concatenate((left_ee, right_ee), axis=0) if robot.has_ee else None
            current_arm_q = robot.get_arm_state()

            action = predict_action(
                observation,
                policy,
                get_safe_torch_device(policy.config.device),
                preprocessor,
                postprocessor,
                policy.config.use_amp,
                step["task"],
                use_dataset=cfg.use_dataset,
                robot_type=None,
            )
            action_np = action.cpu().numpy()

            robot.execute_action(action_np)

            if cfg.save_data:
                process_data_add(
                    sim["episode_writer"],
                    observation,
                    current_arm_q,
                    full_ee_state,
                    action,
                    robot.arm_dof,
                    robot.ee_dof,
                )
                is_success(
                    sim["sim_reward_subscriber"],
                    sim["episode_writer"],
                    sim["reset_pose_publisher"],
                    policy,
                    cfg,
                    reward_stats,
                    init_arm_pose,
                    compat_iface,
                )

            if rerun_logger is not None:
                visualization_data(idx, observation, state_tensor.numpy(), action_np, rerun_logger)
            idx += 1
            reward_stats["episode_num"] = reward_stats["episode_num"] + 1
            time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start_time)))

    except Exception as e:
        logger_mp.info(f"An error occurred: {e}")
    finally:
        for key in ("sim_state_subscriber", "sim_reward_subscriber"):
            sub = sim.get(key)
            if sub is not None:
                try:
                    sub.stop_subscribe()
                    logger_mp.info(f"{key} cleaned up")
                except Exception as e:
                    logger_mp.error(f"Failed to stop {key}: {e}")
        robot.close()


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
