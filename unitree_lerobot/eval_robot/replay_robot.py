"""Replay a recorded dataset on the real robot, using the unified UnitreeRobot."""

import time
import numpy as np

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from unitree_lerobot.eval_robot.robot import UnitreeRobot
from unitree_lerobot.eval_robot.utils.utils import EvalRealConfig
from unitree_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger, visualization_data

import logging_mp

logger_mp = logging_mp.getLogger(__name__)
logger_mp.setLevel(logging_mp.INFO)


@parser.wrap()
def replay_main(cfg: EvalRealConfig):
    logger_mp.info(f"Arguments: {cfg}")

    rerun_logger = RerunLogger() if cfg.visualization else None

    robot = UnitreeRobot(cfg)

    logger_mp.info(f"Starting evaluation loop at {cfg.frequency} Hz.")
    dataset = LeRobotDataset(repo_id=cfg.repo_id, root=cfg.root, episodes=[cfg.episodes])
    actions = dataset.hf_dataset.select_columns("action")

    # init pose
    from_idx = dataset.meta.episodes["dataset_from_index"][0]
    step = dataset[from_idx]
    init_arm_pose = step["observation.state"][: robot.arm_dof].cpu().numpy()

    user_input = input("Please enter the start signal (enter 's' to start the subsequent program):")
    if user_input.lower() != "s":
        return

    logger_mp.info("Initializing robot to starting pose...")
    robot.send_arm(init_arm_pose)
    time.sleep(1)

    try:
        for idx in range(dataset.num_frames):
            loop_start_time = time.perf_counter()
            action_np = actions[idx]["action"].numpy()

            robot.execute_action(action_np, gate_ee_on_nonzero=False)
            logger_mp.info(f"action {action_np}")

            if rerun_logger is not None:
                observation = robot.get_observation()
                state = observation["observation.state"].numpy()
                visualization_data(idx, observation, state, action_np, rerun_logger)

            time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start_time)))
    finally:
        robot.close()


if __name__ == "__main__":
    replay_main()
