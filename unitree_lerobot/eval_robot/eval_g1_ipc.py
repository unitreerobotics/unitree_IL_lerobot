import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO, 
                       file=True, 
                       file_path="/home/unitree/unitree_eai_environment/logs",
                       backup_count=100,
                       max_file_size=50*1024*1024)
logger_mp = logging_mp.getLogger(__name__)

import time
import torch
import numpy as np
import requests
import msgpack
import threading
import msgpack_numpy as m

from PIL import Image
from copy import copy
from multiprocessing.sharedctypes import SynchronizedArray
from sshkeyboard import listen_keyboard, stop_listening

from lerobot.utils.utils import init_logging
from lerobot.configs import parser
from unitree_lerobot.eval_robot.utils.ipc import IPC_Server
from unitree_lerobot.eval_robot.make_robot import (
    setup_image_client,
    setup_robot_interface,
    process_images_and_observations,
)
from unitree_lerobot.eval_robot.utils.utils import (
    cleanup_resources,
    to_list,
    to_scalar,
    EvalRealConfig,
)

m.patch()

# state transition
START = False  # Enable to start robot following VR user motion
STOP = False  # Enable to begin system exit procedure
READY = False  # Ready to (1) enter START state, (2) enter RECORD_RUNNING state
RESET = False  # Enable to reset robot to initial pose


CAMERA_STATUS = False  # camera status
ARM_STATUS = False  # arm status
EE_STATUS = False  # end effector status


def on_press(key):
    global START, RESET, STOP
    if key == "s":
        START = not START
        logger_mp.info(f"==> START = {START}")
    elif key == "r":
        RESET = True
        logger_mp.info("==> RESET = True")
    elif key == "q":
        STOP = True
        logger_mp.info("==> STOP = True")


def get_state() -> dict:
    """Return current heartbeat state"""
    global START, RESET, STOP, READY, CAMERA_STATUS, ARM_STATUS, EE_STATUS
    return {
        "START": START,
        "RESET": RESET,
        "STOP": STOP,
        "READY": READY,
        "CAMERA_STATUS": CAMERA_STATUS,
        "ARM_STATUS": ARM_STATUS,
        "EE_STATUS": EE_STATUS,
    }


class ADBRobotServeClient:
    def __init__(
        self,
        url: str = "http://localhost:8000",
        task: str = "do something.",
        force_predict: bool = False,
    ):
        self.url = url
        self.task = task
        self.force_predict = force_predict
        self.action_buffers: list[np.ndarray] = []

    def predict_action(self, observation: dict) -> np.ndarray:
        """Predict next robot action via HTTP API."""
        if self.force_predict:
            self.action_buffers.clear()

        if not self.action_buffers:
            obs = self._parse_obs(copy(observation))
            actions = self._http_client_call(obs)
            self.action_buffers = list(np.split(actions, actions.shape[0]))

        return self.action_buffers.pop(0).flatten()

    def _resize_image(self, image: np.ndarray, h: int, w: int) -> np.ndarray:
        """Resize image with padding (numpy version)."""
        img = image.transpose(1, 2, 0) if image.ndim == 3 else image
        pil_img = Image.fromarray(img if img.ndim == 3 else np.stack([img] * 3, axis=-1))

        cur_w, cur_h = pil_img.size
        if (cur_w, cur_h) == (w, h):
            return image

        ratio = max(cur_w / w, cur_h / h)
        new_w, new_h = int(cur_w / ratio), int(cur_h / ratio)
        resized = pil_img.resize((new_w, new_h), resample=Image.BILINEAR)

        padded = Image.new("RGB", (w, h), 0)
        pad_x, pad_y = (w - new_w) // 2, (h - new_h) // 2
        padded.paste(resized, (pad_x, pad_y))

        return np.array(padded).transpose(2, 0, 1)

    def _parse_obs(self, obs: dict) -> dict:
        obs["task"] = obs.get("task", self.task)
        for k, v in obs.items():
            if isinstance(v, torch.Tensor):
                v = v.numpy()
                if "images" in k:
                    if v.ndim == 3 and v.shape[0] in [1, 3, 4]:
                        if v.dtype != np.uint8:
                            v = (np.clip(v, 0, 1) * 255).astype(np.uint8)
                        v = self._resize_image(v, 224, 224)
                obs[k] = v
        return obs

    def _http_client_call(self, obs: dict) -> np.ndarray:
        payload = msgpack.packb({"observation": obs}, default=m.encode)
        resp = requests.post(f"{self.url}/act_multi_steps", data=payload)
        if resp.ok:
            return msgpack.unpackb(resp.content, object_hook=m.decode)
        logger_mp.error(f"HTTP {resp.status_code}: {resp.text}")
        return np.array([])


def execute_action(
    action_np: np.ndarray,
    arm_dof: int,
    ee_dof: int,
    arm_ik,
    arm_ctrl,
    ee_shared_mem=None,
    mobile_ctrl=None,
    mobile_action_dim: int = 0,
    base_type: str = "legs",
):
    arm_action = action_np[:arm_dof]
    tau = arm_ik.solve_tau(arm_action)
    arm_ctrl.ctrl_dual_arm(arm_action, tau)

    ee_action_end_idx = arm_dof
    if ee_shared_mem is not None and ee_dof > 0:
        ee_action_start_idx = arm_dof
        ee_action_end_idx = ee_action_start_idx + 2 * ee_dof
        left_ee_action = action_np[ee_action_start_idx : ee_action_start_idx + ee_dof]
        right_ee_action = action_np[ee_action_start_idx + ee_dof : ee_action_start_idx + 2 * ee_dof]

        if isinstance(ee_shared_mem["left"], SynchronizedArray) and np.any(
            np.concatenate((left_ee_action, right_ee_action)) != 0.0
        ):
            ee_shared_mem["left"][:] = to_list(left_ee_action)
            ee_shared_mem["right"][:] = to_list(right_ee_action)
        elif (
            hasattr(ee_shared_mem["left"], "value")
            and hasattr(ee_shared_mem["right"], "value")
            and np.any(np.concatenate((left_ee_action, right_ee_action)) != 0.0)
        ):
            ee_shared_mem["left"].value = to_scalar(left_ee_action)
            ee_shared_mem["right"].value = to_scalar(right_ee_action)

    if mobile_ctrl is not None and mobile_action_dim > 0:
        mobile_action = action_np[ee_action_end_idx : ee_action_end_idx + mobile_action_dim]
        mobile_ctrl.g1_height_action_array_in[0] = float(mobile_action[0])
        if base_type == "mobile_lift" and mobile_action_dim >= 3:
            mobile_ctrl.g1_move_action_array_in[0] = float(mobile_action[1])
            mobile_ctrl.g1_move_action_array_in[1] = float(mobile_action[2])


def get_mobile_state(mobile_ctrl, base_type: str) -> np.ndarray:
    if mobile_ctrl is None or base_type == "legs":
        return np.array([], dtype=np.float64)

    height = float(mobile_ctrl.g1_height_state_array_out[0])
    if base_type == "mobile_lift":
        return np.array(
            [
                height,
                float(mobile_ctrl.g1_move_state_array_out[0]),
                float(mobile_ctrl.g1_move_state_array_out[1]),
            ],
            dtype=np.float64,
        )
    return np.array([height], dtype=np.float64)


@parser.wrap()
def eval_main(cfg: EvalRealConfig):
    try:
        global START, RESET, STOP, READY
        logger_mp.info(cfg)

        policy = ADBRobotServeClient(url=cfg.policy_url, task=cfg.task, force_predict=cfg.force_predict)
        logger_mp.info("ADBRobotServeClient is ok")

        # --- Setup Phase ---
        image_client, image_config = setup_image_client(cfg)
        robot_interface = setup_robot_interface(cfg)

        # fmt: off
        arm_ctrl, arm_ik, ee_shared_mem, arm_dof, ee_dof = (
            robot_interface[key]
            for key in ["arm_ctrl", "arm_ik", "ee_shared_mem", "arm_dof", "ee_dof"]
        )
        mobile_ctrl = robot_interface["mobile_ctrl"]
        mobile_action_dim = int(robot_interface["mobile_action_dim"])
        base_type = getattr(cfg, "base_type", "legs")
        # fmt: on

        idx = 0

        # "The initial positions of the robot's arm and fingers take the initial positions during data recording."
        logger_mp.info("Initializing robot to starting pose...")

        init_pose = cfg.init_pose if cfg.init_pose is not None else np.zeros(arm_dof + 2 * ee_dof + mobile_action_dim)

        execute_action(
            action_np=init_pose,
            arm_dof=arm_dof,
            ee_dof=ee_dof,
            arm_ik=arm_ik,
            arm_ctrl=arm_ctrl,
            ee_shared_mem=ee_shared_mem if cfg.ee else None,
            mobile_ctrl=mobile_ctrl,
            mobile_action_dim=mobile_action_dim,
            base_type=base_type,
        )

        time.sleep(1.0)  # Give time for the robot to move

        # --- Run Main Loop ---
        logger_mp.info(f"Starting evaluation loop at {cfg.frequency} Hz.")

        if cfg.ipc:
            ipc_server = IPC_Server(on_press=on_press, get_state=get_state)
            ipc_server.start()

        else:
            listen_keyboard_thread = threading.Thread(
                target=listen_keyboard,
                kwargs={
                    "on_press": on_press,
                    "until": None,
                    "sequential": False,
                },
                daemon=True,
            )
            listen_keyboard_thread.start()

        logger_mp.info("Please enter the start signal (enter 's' to start/stop the subsequent program)")
        READY = True  # Now ready to accept START command
        test_start_time = time.perf_counter()

        while (not STOP) and READY:
            loop_start_time = time.perf_counter()

            # 1. Get Observations
            observation, current_arm_q, _ = process_images_and_observations(image_client, image_config, arm_ctrl)
            try:
                left_ee_state = right_ee_state = np.array([])
                if cfg.ee:
                    with ee_shared_mem["lock"]:
                        full_state = np.array(ee_shared_mem["state"][:])
                        left_ee_state = full_state[:ee_dof]
                        right_ee_state = full_state[ee_dof:]
                        EE_STATUS = True
            except Exception as e:
                logger_mp.error(f"[process_images_and_observations] Failed to get arm state: {e}")
                left_ee_state = right_ee_state = None
                EE_STATUS = False

            mobile_state = get_mobile_state(mobile_ctrl, base_type)
            state = np.concatenate((current_arm_q, left_ee_state, right_ee_state, mobile_state), axis=0)
            observation["observation.state"] = torch.from_numpy(state).float()
            observation["task"] = cfg.task if cfg.task else None

            if RESET:
                # 1️⃣ Reset phase: interpolate from current position to initial pose
                logger_mp.info("Resetting robot to initial pose...")
                interp_poses = np.linspace(state, init_pose, 30)
                for q in interp_poses:
                    execute_action(
                        action_np=q,
                        arm_dof=arm_dof,
                        ee_dof=ee_dof,
                        arm_ik=arm_ik,
                        arm_ctrl=arm_ctrl,
                        ee_shared_mem=ee_shared_mem if cfg.ee else None,
                        mobile_ctrl=mobile_ctrl,
                        mobile_action_dim=mobile_action_dim,
                        base_type=base_type,
                    )
                    time.sleep(1.0 / cfg.frequency)
                logger_mp.info("Reset complete.")
                RESET = False
                START = False
                action_np = init_pose.copy()  # keep final position at initial pose

            elif START:
                # 2️⃣ START phase: model inference
                action_np = policy.predict_action(observation)
            else:
                # 3️⃣ Hold current position
                action_np = state.copy()
            # 3. Execute the action
            execute_action(
                action_np=action_np,
                arm_dof=arm_dof,
                ee_dof=ee_dof,
                arm_ik=arm_ik,
                arm_ctrl=arm_ctrl,
                ee_shared_mem=ee_shared_mem if cfg.ee else None,
                mobile_ctrl=mobile_ctrl,
                mobile_action_dim=mobile_action_dim,
                base_type=base_type,
            )
            idx += 1
            time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start_time)))

    except KeyboardInterrupt:
        logger_mp.info("KeyboardInterrupt, exiting program...")
    finally:
        # arm go home
        try:
            arm_ctrl.ctrl_dual_arm_go_home()
        except Exception as e:
            logger_mp.error(f"Failed to ctrl_dual_arm_go_home: {e}")

        # stop keyboard listener or ipc server
        try:
            if cfg.ipc:
                ipc_server.stop()
            else:
                stop_listening()
                listen_keyboard_thread.join()
        except Exception as e:
            logger_mp.error(f"Failed to stop keyboard listener or ipc server: {e}")

        logger_mp.info("Finally, exiting program.")
        exit(0)


if __name__ == "__main__":
    init_logging()
    eval_main()
