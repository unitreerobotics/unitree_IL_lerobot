"""Hardware test harness using the unified UnitreeRobot (no policy server).

python -m unitree_lerobot.eval_robot.eval_g1_ipc_test \
  --arm G1_29 \
  --ee dex3 \
  --base-type legs \
  --hardware-test-joint 0 \
  --hardware-test-amplitude 0.08 \
  --hardware-test-period 5.0
"""

import time
import threading

import logging_mp
import numpy as np
from sshkeyboard import listen_keyboard, stop_listening

from lerobot.configs import parser
from lerobot.utils.utils import init_logging

from unitree_lerobot.eval_robot.robot import UnitreeRobot
from unitree_lerobot.eval_robot.utils.ipc import IPC_Server
from unitree_lerobot.eval_robot.utils.utils import EvalRealConfig

logger_mp = logging_mp.getLogger(__name__)
logger_mp.setLevel(logging_mp.INFO)

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


def _build_hold_action(robot: UnitreeRobot, state: np.ndarray) -> np.ndarray:
    """Build an action that holds the current pose; mobile velocity is zeroed."""
    hold = state.copy()
    if robot.base_type == "mobile_lift" and robot.mobile_action_dim >= 3:
        # Keep the height, but send zero velocity commands.
        hold[-2] = 0.0
        hold[-1] = 0.0
    return hold


def _generate_hw_test_action(
    cfg: EvalRealConfig,
    robot: UnitreeRobot,
    state: np.ndarray,
    elapsed_s: float,
) -> np.ndarray:
    action = state.copy()
    joint_idx = int(np.clip(cfg.hardware_test_joint, 0, robot.arm_dof - 1))
    period = max(float(cfg.hardware_test_period), 1e-3)
    amplitude = float(cfg.hardware_test_amplitude)
    action[joint_idx] = state[joint_idx] + amplitude * np.sin(2.0 * np.pi * elapsed_s / period)

    if robot.mobile_action_dim > 0:
        mobile_start = robot.arm_dof + 2 * robot.ee_dof
        action[mobile_start] = state[mobile_start] + float(cfg.hardware_test_lift_delta)
        if robot.base_type == "mobile_lift" and robot.mobile_action_dim >= 3:
            action[mobile_start + 1] = float(cfg.hardware_test_move_x)
            action[mobile_start + 2] = float(cfg.hardware_test_move_yaw)
    return action


@parser.wrap()
def eval_main(cfg: EvalRealConfig):
    global START, STOP, READY, RESET, EE_STATUS
    ipc_server = None
    listen_keyboard_thread = None
    robot: UnitreeRobot | None = None

    try:
        logger_mp.info(cfg)
        logger_mp.info("Initializing hardware test mode, skip policy server.")

        robot = UnitreeRobot(cfg)

        # Initial pose
        if cfg.init_pose is not None:
            init_pose = np.asarray(cfg.init_pose, dtype=np.float64)
        else:
            # Zero arm + EE, keep current mobile height
            init_pose = np.zeros(robot.action_dim, dtype=np.float64)
            if robot.mobile_action_dim > 0:
                init_pose[-robot.mobile_action_dim :] = robot.get_mobile_state()

        if init_pose.shape[0] != robot.action_dim:
            raise ValueError(f"init_pose dim {init_pose.shape[0]} != action_dim {robot.action_dim}")

        robot.execute_action(init_pose, gate_ee_on_nonzero=False)
        time.sleep(1.0)

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
        test_start_time = time.perf_counter()

        while (not STOP) and READY:
            loop_start_time = time.perf_counter()

            state = robot.get_state()
            EE_STATUS = robot.has_ee
            logger_mp.debug(f"state={state.tolist()}")

            if RESET:
                logger_mp.info("Resetting robot to initial pose...")
                robot.reset_to(init_pose, steps=30)
                logger_mp.info("Reset complete.")
                RESET = False
                START = False
                continue
            elif START:
                action_np = _generate_hw_test_action(cfg, robot, state, loop_start_time - test_start_time)
            else:
                action_np = _build_hold_action(robot, state)

            robot.execute_action(action_np, gate_ee_on_nonzero=False)
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
