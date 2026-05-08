from multiprocessing import shared_memory, Value, Array, Lock
from typing import Any
import numpy as np
import argparse
import threading
import torch
import cv2
from unitree_lerobot.eval_robot.image_server.image_client import ImageClient
from unitree_lerobot.eval_robot.robot_control.robot_arm import (
    G1_29_ArmController,
    G1_23_ArmController,
)
from unitree_lerobot.eval_robot.robot_control.robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK
from unitree_lerobot.eval_robot.robot_control.robot_hand_unitree import (
    Dex3_1_Controller,
    Dex1_1_Gripper_Controller,
)

from unitree_lerobot.eval_robot.utils.episode_writer import EpisodeWriter

from unitree_lerobot.eval_robot.robot_control.robot_hand_inspire import Inspire_Controller
from unitree_lerobot.eval_robot.robot_control.robot_hand_brainco import Brainco_Controller


from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

import logging_mp

logger_mp = logging_mp.getLogger(__name__)
logger_mp.setLevel(logging_mp.INFO)

# Configuration for robot arms
ARM_CONFIG = {
    "G1_29": {"controller": G1_29_ArmController, "ik_solver": G1_29_ArmIK, "dof": 14},
    "G1_23": {"controller": G1_23_ArmController, "ik_solver": G1_23_ArmIK, "dof": 14},
    # Add other arms here
}

# Configuration for end-effectors
EE_CONFIG: dict[str, dict[str, Any]] = {
    "dex3": {
        "controller": Dex3_1_Controller,
        "dof": 7,
        "shared_mem_type": "Array",
        "shared_mem_size": 7,
        # "out_len": 14,
    },
    "dex1": {
        "controller": Dex1_1_Gripper_Controller,
        "dof": 1,
        "shared_mem_type": "Value",
        # "out_len": 2,
    },
    "inspire1": {
        "controller": Inspire_Controller,
        "dof": 6,
        "shared_mem_type": "Array",
        "shared_mem_size": 6,
        # "out_len": 12,
    },
    "brainco": {
        "controller": Brainco_Controller,
        "dof": 6,
        "shared_mem_type": "Array",
        "shared_mem_size": 6,
        # "out_len": 12,
    },
}


def setup_image_client(args: argparse.Namespace) -> dict[str, Any]:
    """Initializes and starts the image client and shared memory."""
    # image client: img_config should be the same as the configuration in image_server.py (of Robot's development computing unit)
    
    image_client = ImageClient(host=args.image_host)
    image_config = image_client.get_cam_config()
    return image_client, image_config


def _resolve_out_len(spec: dict[str, Any]) -> int:
    return int(spec.get("out_len", 2 * int(spec["dof"])))


def setup_robot_interface(args: argparse.Namespace) -> dict[str, Any]:
    """
    Initializes robot controllers and IK solvers based on configuration.
    """
    # ---------- Arm ----------
    arm_spec = ARM_CONFIG[args.arm]
    arm_ik = arm_spec["ik_solver"]()
    is_sim = getattr(args, "sim", False)
    arm_ctrl = arm_spec["controller"](motion_mode=args.motion, simulation_mode=is_sim)

    # ---------- End Effector (optional) ----------
    ee_ctrl, ee_shared_mem, ee_dof = None, {}, 0

    if ee_key := getattr(args, "ee", "").lower():
        if ee_key not in EE_CONFIG:
            raise ValueError(f"Unknown end-effector '{args.ee}'. Available: {list(EE_CONFIG.keys())}")

        spec = EE_CONFIG[ee_key]
        mem_type, out_len, ee_dof = spec["shared_mem_type"].lower(), _resolve_out_len(spec), spec["dof"]
        data_lock = Lock()

        left_in, right_in = (
            (Array("d", spec["shared_mem_size"], lock=True), Array("d", spec["shared_mem_size"], lock=True))
            if mem_type == "array"
            else (Value("d", 0.0, lock=True), Value("d", 0.0, lock=True))
        )

        state_arr, action_arr = Array("d", out_len, lock=False), Array("d", out_len, lock=False)

        ee_ctrl = spec["controller"](left_in, right_in, data_lock, state_arr, action_arr, simulation_mode=is_sim)

        ee_shared_mem = {
            "left": left_in,
            "right": right_in,
            "state": state_arr,
            "action": action_arr,
            "lock": data_lock,
        }

    # ---------- Simulation helpers (optional) ----------
    episode_writer = None
    if is_sim:
        reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
        reset_pose_publisher.Init()
        from unitree_lerobot.eval_robot.utils.sim_state_topic import (
            start_sim_state_subscribe,
            start_sim_reward_subscribe,
        )

        sim_state_subscriber = start_sim_state_subscribe()
        sim_reward_subscriber = start_sim_reward_subscribe()
        if getattr(args, "save_data", False) and getattr(args, "task_dir", None):
            episode_writer = EpisodeWriter(args.task_dir, frequency=30, image_size=[640, 480])
        return {
            "arm_ctrl": arm_ctrl,
            "arm_ik": arm_ik,
            "ee_ctrl": ee_ctrl,
            "ee_shared_mem": ee_shared_mem,
            "arm_dof": int(arm_spec["dof"]),
            "ee_dof": ee_dof,
            "sim_state_subscriber": sim_state_subscriber,
            "sim_reward_subscriber": sim_reward_subscriber,
            "episode_writer": episode_writer,
            "reset_pose_publisher": reset_pose_publisher,
        }
    return {
        "arm_ctrl": arm_ctrl,
        "arm_ik": arm_ik,
        "ee_ctrl": ee_ctrl,
        "ee_shared_mem": ee_shared_mem,
        "arm_dof": int(arm_spec["dof"]),
        "ee_dof": ee_dof,
    }


def process_images_and_observations(img_client, camera_config, arm_ctrl):
    status = {"image_ok": False, "arm_ok": False}
    try:
        """Processes images and generates observations."""
        observation = {}
        def to_tensor_rgb(img):
            return torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if camera_config['head_camera']['enable_zmq']:
            head_img, head_img_fps = img_client.get_head_frame()
            if head_img is not None:
                observation["observation.images.cam_left_high"] = to_tensor_rgb(head_img[:, :camera_config['head_camera']['image_shape'][1]//2])
                observation["observation.images.cam_right_high"] = to_tensor_rgb(head_img[:, camera_config['head_camera']['image_shape'][1]//2:])

        if camera_config['left_wrist_camera']['enable_zmq']:
            left_wrist, _ = img_client.get_left_wrist_frame()
            if left_wrist is not None:
                observation["observation.images.cam_left_wrist"] = to_tensor_rgb(left_wrist)
        if camera_config['right_wrist_camera']['enable_zmq']:
            right_wrist, _ = img_client.get_right_wrist_frame()
            if right_wrist is not None:
                observation["observation.images.cam_right_wrist"] = to_tensor_rgb(right_wrist)

        status["image_ok"] = True

    except Exception as e:
        logger_mp.error(f"[process_images_and_observations] Failed to process images: {e}")
        observation = {
            "observation.images.cam_left_high": None,
            "observation.images.cam_right_high": None,
            "observation.images.cam_left_wrist": None,
            "observation.images.cam_right_wrist": None,
        }
    try:
        current_arm_q = arm_ctrl.get_current_dual_arm_q()
        status["arm_ok"] = True
    except Exception as e:
        logger_mp.error(f"[process_images_and_observations] Failed to get arm state: {e}")
        current_arm_q = None

    return observation, current_arm_q, status


def publish_reset_category(category: int, publisher):  # Scene Reset signal
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")
