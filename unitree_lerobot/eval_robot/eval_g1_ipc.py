"""Policy-server driven evaluation loop (IPC or keyboard controlled) using UnitreeRobot."""

import logging_mp

logging_mp.basicConfig(
    level=logging_mp.INFO,
    file=True,
    file_path="/home/unitree/unitree_eai_environment/logs",
    backup_count=100,
    max_file_size=50 * 1024 * 1024,
)
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
from sshkeyboard import listen_keyboard, stop_listening

from lerobot.utils.utils import init_logging
from lerobot.configs import parser

from unitree_lerobot.eval_robot.robot import UnitreeRobot
from unitree_lerobot.eval_robot.utils.ipc import IPC_Server
from unitree_lerobot.eval_robot.utils.utils import EvalRealConfig

m.patch()

# ----- state flags -----
START = False
STOP = False
READY = False
RESET = False

CAMERA_STATUS = False
ARM_STATUS = False
EE_STATUS = False


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
    def __init__(self, url: str = "http://localhost:8000", task: str = "do something.", force_predict: bool = False):
        self.url = url
        self.task = task
        self.force_predict = force_predict
        self.action_buffers: list[np.ndarray] = []

    def predict_action(self, observation: dict) -> np.ndarray:
        if self.force_predict:
            self.action_buffers.clear()
        if not self.action_buffers:
            obs = self._parse_obs(copy(observation))
            actions = self._http_client_call(obs)
            self.action_buffers = list(np.split(actions, actions.shape[0]))
        return self.action_buffers.pop(0).flatten()

    def _resize_image(self, image: np.ndarray, h: int, w: int) -> np.ndarray:
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


def _default_init_pose(robot: UnitreeRobot) -> np.ndarray:
    """If the user did not supply init_pose, hold the current mobile height."""
    pose = np.zeros(robot.action_dim, dtype=np.float64)
    if robot.mobile_action_dim > 0:
        mobile = robot.get_mobile_state()
        pose[-robot.mobile_action_dim :] = mobile
    return pose


@parser.wrap()
def eval_main(cfg: EvalRealConfig):
    global START, RESET, STOP, READY
    ipc_server = None
    listen_keyboard_thread = None
    robot: UnitreeRobot | None = None

    try:
        logger_mp.info(cfg)

        policy = ADBRobotServeClient(url=cfg.policy_url, task=cfg.task, force_predict=cfg.force_predict)
        logger_mp.info("ADBRobotServeClient is ok")

        robot = UnitreeRobot(cfg)

        init_pose = np.asarray(cfg.init_pose, dtype=np.float64) if cfg.init_pose is not None else _default_init_pose(robot)
        if init_pose.shape[0] != robot.action_dim:
            raise ValueError(f"init_pose dim {init_pose.shape[0]} != action_dim {robot.action_dim}")

        logger_mp.info("Initializing robot to starting pose...")
        robot.execute_action(init_pose, gate_ee_on_nonzero=False)
        time.sleep(1.0)

        logger_mp.info(f"Starting evaluation loop at {cfg.frequency} Hz.")

        if cfg.ipc:
            ipc_server = IPC_Server(on_press=on_press, get_state=get_state)
            ipc_server.start()
        else:
            listen_keyboard_thread = threading.Thread(
                target=listen_keyboard,
                kwargs={"on_press": on_press, "until": None, "sequential": False},
                daemon=True,
            )
            listen_keyboard_thread.start()

        logger_mp.info("Please enter the start signal (enter 's' to start/stop the subsequent program)")
        READY = True

        while (not STOP) and READY:
            loop_start_time = time.perf_counter()

            observation = robot.get_observation(task=cfg.task or None)
            state = observation["observation.state"].numpy()

            if RESET:
                logger_mp.info("Resetting robot to initial pose...")
                robot.reset_to(init_pose, steps=30)
                logger_mp.info("Reset complete.")
                RESET = False
                START = False
                action_np = init_pose.copy()
            elif START:
                action_np = policy.predict_action(observation)
            else:
                action_np = state.copy()

            robot.execute_action(action_np, gate_ee_on_nonzero=True)
            time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start_time)))

    except KeyboardInterrupt:
        logger_mp.info("KeyboardInterrupt, exiting program...")
    finally:
        if robot is not None:
            robot.go_home()
        try:
            if ipc_server is not None:
                ipc_server.stop()
            elif listen_keyboard_thread is not None:
                stop_listening()
                listen_keyboard_thread.join()
        except Exception as e:
            logger_mp.error(f"Failed to stop keyboard listener or ipc server: {e}")

        if robot is not None:
            robot.close()
        logger_mp.info("Finally, exiting program.")
        exit(0)


if __name__ == "__main__":
    init_logging()
    eval_main()
