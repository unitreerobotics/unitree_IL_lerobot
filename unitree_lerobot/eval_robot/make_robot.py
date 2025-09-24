from multiprocessing import shared_memory, Value, Array, Lock
from typing import Any, Dict
import numpy as np
import argparse
import threading
import torch
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

logging_mp.basic_config(level=logging_mp.INFO)
logger_mp = logging_mp.get_logger(__name__)

# Configuration for robot arms
ARM_CONFIG = {
    "G1_29": {"controller": G1_29_ArmController, "ik_solver": G1_29_ArmIK, "dof": 14},
    "G1_23": {"controller": G1_23_ArmController, "ik_solver": G1_23_ArmIK, "dof": 14},
    # Add other arms here
}

# Configuration for end-effectors
EE_CONFIG: Dict[str, Dict[str, Any]] = {
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


def setup_image_client(args: argparse.Namespace) -> Dict[str, Any]:
    """Initializes and starts the image client and shared memory."""
    # image client: img_config should be the same as the configuration in image_server.py (of Robot's development computing unit)
    if getattr(args, "sim", False):
        img_config = {
            "fps": 30,
            "head_camera_type": "opencv",
            "head_camera_image_shape": [480, 640],  # Head camera resolution
            "head_camera_id_numbers": [0],
            "wrist_camera_type": "opencv",
            "wrist_camera_image_shape": [480, 640],  # Wrist camera resolution
            "wrist_camera_id_numbers": [2, 4],
        }
    else:
        img_config = {
            "fps": 30,
            "head_camera_type": "opencv",
            "head_camera_image_shape": [480, 1280],  # Head camera resolution
            "head_camera_id_numbers": [0],
            "wrist_camera_type": "opencv",
            "wrist_camera_image_shape": [480, 640],  # Wrist camera resolution
            "wrist_camera_id_numbers": [2, 4],
        }

    ASPECT_RATIO_THRESHOLD = 2.0  # If the aspect ratio exceeds this value, it is considered binocular
    if len(img_config["head_camera_id_numbers"]) > 1 or (
        img_config["head_camera_image_shape"][1] / img_config["head_camera_image_shape"][0] > ASPECT_RATIO_THRESHOLD
    ):
        BINOCULAR = True
    else:
        BINOCULAR = False
    if "wrist_camera_type" in img_config:
        WRIST = True
    else:
        WRIST = False

    if BINOCULAR and not (
        img_config["head_camera_image_shape"][1] / img_config["head_camera_image_shape"][0] > ASPECT_RATIO_THRESHOLD
    ):
        tv_img_shape = (img_config["head_camera_image_shape"][0], img_config["head_camera_image_shape"][1] * 2, 3)
    else:
        tv_img_shape = (img_config["head_camera_image_shape"][0], img_config["head_camera_image_shape"][1], 3)

    tv_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(tv_img_shape) * np.uint8().itemsize)
    tv_img_array = np.ndarray(tv_img_shape, dtype=np.uint8, buffer=tv_img_shm.buf)

    if WRIST and getattr(args, "sim", False):
        wrist_img_shape = (img_config["wrist_camera_image_shape"][0], img_config["wrist_camera_image_shape"][1] * 2, 3)
        wrist_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(wrist_img_shape) * np.uint8().itemsize)
        wrist_img_array = np.ndarray(wrist_img_shape, dtype=np.uint8, buffer=wrist_img_shm.buf)
        img_client = ImageClient(
            tv_img_shape=tv_img_shape,
            tv_img_shm_name=tv_img_shm.name,
            wrist_img_shape=wrist_img_shape,
            wrist_img_shm_name=wrist_img_shm.name,
            server_address="127.0.0.1",
        )
    elif WRIST and not getattr(args, "sim", False):
        wrist_img_shape = (img_config["wrist_camera_image_shape"][0], img_config["wrist_camera_image_shape"][1] * 2, 3)
        wrist_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(wrist_img_shape) * np.uint8().itemsize)
        wrist_img_array = np.ndarray(wrist_img_shape, dtype=np.uint8, buffer=wrist_img_shm.buf)
        img_client = ImageClient(
            tv_img_shape=tv_img_shape,
            tv_img_shm_name=tv_img_shm.name,
            wrist_img_shape=wrist_img_shape,
            wrist_img_shm_name=wrist_img_shm.name,
        )
    else:
        img_client = ImageClient(tv_img_shape=tv_img_shape, tv_img_shm_name=tv_img_shm.name)

    has_wrist_cam = "wrist_camera_type" in img_config

    image_receive_thread = threading.Thread(target=img_client.receive_process, daemon=True)
    image_receive_thread.daemon = True
    image_receive_thread.start()

    return {
        "tv_img_array": tv_img_array,
        "wrist_img_array": wrist_img_array,
        "tv_img_shape": tv_img_shape,
        "wrist_img_shape": wrist_img_shape,
        "is_binocular": BINOCULAR,
        "has_wrist_cam": has_wrist_cam,
        "shm_resources": [tv_img_shm, wrist_img_shm],
    }


def _resolve_out_len(spec: Dict[str, Any]) -> int:
    return int(spec.get("out_len", 2 * int(spec["dof"])))


def setup_robot_interface(args: argparse.Namespace) -> Dict[str, Any]:
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
        from unitree_lerobot.eval_robot.utils.sim_state_topic import start_sim_state_subscribe, start_sim_reward_subscribe

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

def process_images_and_observations(
    tv_img_array, wrist_img_array, tv_img_shape, wrist_img_shape, is_binocular, has_wrist_cam, arm_ctrl
):
    """Processes images and generates observations."""
    current_tv_image = tv_img_array.copy()
    current_wrist_image = wrist_img_array.copy() if has_wrist_cam else None

    left_top_cam = current_tv_image[:, : tv_img_shape[1] // 2] if is_binocular else current_tv_image
    right_top_cam = current_tv_image[:, tv_img_shape[1] // 2 :] if is_binocular else None

    left_wrist_cam = right_wrist_cam = None
    if has_wrist_cam and current_wrist_image is not None:
        left_wrist_cam = current_wrist_image[:, : wrist_img_shape[1] // 2]
        right_wrist_cam = current_wrist_image[:, wrist_img_shape[1] // 2 :]
    observation = {
        "observation.images.cam_left_high": torch.from_numpy(left_top_cam),
        "observation.images.cam_right_high": torch.from_numpy(right_top_cam) if is_binocular else None,
        "observation.images.cam_left_wrist": torch.from_numpy(left_wrist_cam) if has_wrist_cam else None,
        "observation.images.cam_right_wrist": torch.from_numpy(right_wrist_cam) if has_wrist_cam  else None,
    }
    current_arm_q = arm_ctrl.get_current_dual_arm_q()

    return observation, current_arm_q


def publish_reset_category(category: int,publisher): # Scene Reset signal
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")