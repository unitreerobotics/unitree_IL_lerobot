"""
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
from multiprocessing.sharedctypes import SynchronizedArray
from sshkeyboard import listen_keyboard, stop_listening

from lerobot.configs import parser
from lerobot.utils.utils import init_logging
from unitree_lerobot.eval_robot.make_robot import (
    process_images_and_observations,
    setup_image_client,
    setup_robot_interface,
)
from unitree_lerobot.eval_robot.utils.ipc import IPC_Server
from unitree_lerobot.eval_robot.utils.utils import EvalRealConfig, to_list, to_scalar

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


def build_hold_action(
    arm_state: np.ndarray,
    left_ee_state: np.ndarray,
    right_ee_state: np.ndarray,
    mobile_state: np.ndarray,
    base_type: str,
) -> np.ndarray:
    chunks = [arm_state, left_ee_state, right_ee_state]
    if mobile_state.size > 0:
        if base_type == "mobile_lift":
            chunks.append(np.array([mobile_state[0], 0.0, 0.0], dtype=np.float64))
        else:
            chunks.append(np.array([mobile_state[0]], dtype=np.float64))
    return np.concatenate(chunks, axis=0)


def validate_action_dim(action_np: np.ndarray, expected_dim: int, source: str):
    if action_np.shape[0] != expected_dim:
        raise ValueError(f"{source} dim mismatch: expected {expected_dim}, got {action_np.shape[0]}")


def generate_hardware_test_action(
    cfg: EvalRealConfig,
    current_arm_q: np.ndarray,
    left_ee_state: np.ndarray,
    right_ee_state: np.ndarray,
    mobile_state: np.ndarray,
    arm_dof: int,
    base_type: str,
    elapsed_s: float,
) -> np.ndarray:
    arm_action = np.array(current_arm_q, dtype=np.float64, copy=True)
    joint_idx = int(np.clip(cfg.hardware_test_joint, 0, arm_dof - 1))
    period = max(float(cfg.hardware_test_period), 1e-3)
    amplitude = float(cfg.hardware_test_amplitude)
    arm_action[joint_idx] = current_arm_q[joint_idx] + amplitude * np.sin(2.0 * np.pi * elapsed_s / period)

    chunks = [
        arm_action,
        np.array(left_ee_state, dtype=np.float64, copy=True),
        np.array(right_ee_state, dtype=np.float64, copy=True),
    ]

    if mobile_state.size > 0:
        lift_target = float(mobile_state[0] + cfg.hardware_test_lift_delta)
        if base_type == "mobile_lift":
            chunks.append(
                np.array(
                    [lift_target, float(cfg.hardware_test_move_x), float(cfg.hardware_test_move_yaw)],
                    dtype=np.float64,
                )
            )
        else:
            chunks.append(np.array([lift_target], dtype=np.float64))

    return np.concatenate(chunks, axis=0)


@parser.wrap()
def eval_main(cfg: EvalRealConfig):
    try:
        global START, STOP, READY, RESET, EE_STATUS
        logger_mp.info(cfg)
        logger_mp.info("Initializing hardware test mode, skip policy server.")

        image_client, image_config = setup_image_client(cfg)
        robot_interface = setup_robot_interface(cfg)

        arm_ctrl, arm_ik, ee_shared_mem, arm_dof, ee_dof = (
            robot_interface[key] for key in ["arm_ctrl", "arm_ik", "ee_shared_mem", "arm_dof", "ee_dof"]
        )
        mobile_ctrl = robot_interface["mobile_ctrl"]
        mobile_action_dim = int(robot_interface["mobile_action_dim"])
        base_type = getattr(cfg, "base_type", "legs")

        init_pose = cfg.init_pose if cfg.init_pose is not None else None
        current_mobile_state = get_mobile_state(mobile_ctrl, base_type)
        if init_pose is None:
            init_pose = np.concatenate(
                (
                    np.zeros(arm_dof + 2 * ee_dof, dtype=np.float64),
                    build_hold_action(
                        arm_state=np.array([], dtype=np.float64),
                        left_ee_state=np.array([], dtype=np.float64),
                        right_ee_state=np.array([], dtype=np.float64),
                        mobile_state=current_mobile_state,
                        base_type=base_type,
                    ),
                ),
                axis=0,
            )
        init_pose = np.asarray(init_pose, dtype=np.float64)
        expected_action_dim = arm_dof + 2 * ee_dof + mobile_action_dim
        validate_action_dim(init_pose, expected_action_dim, "init_pose")

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

            _, current_arm_q, _ = process_images_and_observations(image_client, image_config, arm_ctrl)

            left_ee_state = right_ee_state = np.array([], dtype=np.float64)
            try:
                if cfg.ee:
                    with ee_shared_mem["lock"]:
                        full_state = np.array(ee_shared_mem["state"][:], dtype=np.float64)
                        left_ee_state = full_state[:ee_dof]
                        right_ee_state = full_state[ee_dof:]
                        EE_STATUS = True
            except Exception as e:
                logger_mp.error(f"[eval_g1_ipc_test] Failed to get end-effector state: {e}")
                left_ee_state = right_ee_state = np.array([], dtype=np.float64)
                EE_STATUS = False

            mobile_state = get_mobile_state(mobile_ctrl, base_type)
            state = np.concatenate((current_arm_q, left_ee_state, right_ee_state, mobile_state), axis=0)
            logger_mp.debug(f"state={state.tolist()}")

            if RESET:
                logger_mp.info("Resetting robot to initial pose...")
                reset_start = build_hold_action(
                    arm_state=current_arm_q,
                    left_ee_state=left_ee_state,
                    right_ee_state=right_ee_state,
                    mobile_state=mobile_state,
                    base_type=base_type,
                )
                interp_poses = np.linspace(reset_start, init_pose, 30)
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
                action_np = init_pose.copy()
            elif START:
                action_np = generate_hardware_test_action(
                    cfg=cfg,
                    current_arm_q=current_arm_q,
                    left_ee_state=left_ee_state,
                    right_ee_state=right_ee_state,
                    mobile_state=mobile_state,
                    arm_dof=arm_dof,
                    base_type=base_type,
                    elapsed_s=loop_start_time - test_start_time,
                )
            else:
                action_np = build_hold_action(
                    arm_state=current_arm_q,
                    left_ee_state=left_ee_state,
                    right_ee_state=right_ee_state,
                    mobile_state=mobile_state,
                    base_type=base_type,
                )

            validate_action_dim(action_np, expected_action_dim, "execute action")
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
            time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start_time)))

    except KeyboardInterrupt:
        logger_mp.info("KeyboardInterrupt, exiting program...")
    finally:
        try:
            arm_ctrl.ctrl_dual_arm_go_home()
        except Exception as e:
            logger_mp.error(f"Failed to ctrl_dual_arm_go_home: {e}")

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
