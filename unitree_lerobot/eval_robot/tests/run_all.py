"""One-shot read-only self-check: exercises arm + ee + camera + mobile base
in sequence, sending no motion commands. Used to verify hardware / DDS
connectivity before running policy or motion tests.

python -m unitree_lerobot.eval_robot.tests.run_all \
    --arm G1_29 --ee dex3 --base-type only_height \
    --image-host 192.168.123.164

Each component is optional — omit the flag to skip that subsystem.
"""

from __future__ import annotations

import argparse
import time
import traceback
from dataclasses import dataclass

import numpy as np

import logging_mp

logger_mp = logging_mp.getLogger(__name__)
logger_mp.setLevel(logging_mp.INFO)


@dataclass
class Result:
    name: str
    ok: bool
    detail: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", default=None, choices=[None, "G1_29", "G1_23"], nargs="?")
    p.add_argument("--ee", default=None, choices=[None, "dex3", "dex1", "inspire1", "brainco"], nargs="?")
    p.add_argument("--base-type", default=None, choices=[None, "legs", "only_height", "mobile_lift"], nargs="?")
    p.add_argument("--image-host", default=None)
    p.add_argument("--sim", action="store_true")
    p.add_argument("--ee-wait", type=float, default=1.0, help="Seconds to let EE background process warm up")
    return p.parse_args()


def check_arm(arm_key: str, sim: bool) -> Result:
    from unitree_lerobot.eval_robot.robot_control.robot_arm import (
        G1_23_ArmController,
        G1_29_ArmController,
    )

    ctrls = {"G1_29": G1_29_ArmController, "G1_23": G1_23_ArmController}
    try:
        ctrl = ctrls[arm_key](motion_mode=False, simulation_mode=sim)
        q = np.asarray(ctrl.get_current_dual_arm_q(), dtype=np.float64)
        return Result("arm", True, f"dof={len(q)} q[:3]={q[:3].tolist()}")
    except Exception as e:
        return Result("arm", False, f"{type(e).__name__}: {e}")


def check_ee(ee_key: str, sim: bool, wait_s: float) -> Result:
    from multiprocessing import Array, Lock, Value

    from unitree_lerobot.eval_robot.robot_control.robot_hand_brainco import Brainco_Controller
    from unitree_lerobot.eval_robot.robot_control.robot_hand_inspire import Inspire_Controller
    from unitree_lerobot.eval_robot.robot_control.robot_hand_unitree import (
        Dex1_1_Gripper_Controller,
        Dex3_1_Controller,
    )

    registry = {
        "dex3":     (Dex3_1_Controller,         7, "array", 7),
        "dex1":     (Dex1_1_Gripper_Controller, 1, "value", 0),
        "inspire1": (Inspire_Controller,        6, "array", 6),
        "brainco":  (Brainco_Controller,        6, "array", 6),
    }
    try:
        Ctrl, dof, mem_type, mem_size = registry[ee_key]
        lock = Lock()
        if mem_type == "array":
            left = Array("d", mem_size, lock=True)
            right = Array("d", mem_size, lock=True)
        else:
            left = Value("d", 0.0, lock=True)
            right = Value("d", 0.0, lock=True)
        state = Array("d", 2 * dof, lock=False)
        action = Array("d", 2 * dof, lock=False)
        _ = Ctrl(left, right, lock, state, action, simulation_mode=sim)
        time.sleep(max(wait_s, 0.0))
        with lock:
            s = np.array(state[:], dtype=np.float64)
        return Result("ee", True, f"key={ee_key} dof={dof} state[:dof]={s[:dof].tolist()}")
    except Exception as e:
        return Result("ee", False, f"{type(e).__name__}: {e}")


def check_camera(image_host: str) -> Result:
    from unitree_lerobot.eval_robot.image_server.image_client import ImageClient

    try:
        client = ImageClient(host=image_host, request_bgr=True)
        cfg = client.get_cam_config()
        enabled = {
            "head": cfg.get("head_camera", {}).get("enable_zmq", False),
            "left_wrist": cfg.get("left_wrist_camera", {}).get("enable_zmq", False),
            "right_wrist": cfg.get("right_wrist_camera", {}).get("enable_zmq", False),
        }
        got: dict[str, bool] = {}
        getters = {
            "head": client.get_head_frame,
            "left_wrist": client.get_left_wrist_frame,
            "right_wrist": client.get_right_wrist_frame,
        }
        deadline = time.perf_counter() + 2.0
        for name, on in enabled.items():
            if not on:
                continue
            got[name] = False
            while time.perf_counter() < deadline and not got[name]:
                f = getters[name]()
                if f is not None and getattr(f, "bgr", None) is not None:
                    got[name] = True
                    break
                time.sleep(0.05)
        missing = [n for n, v in got.items() if not v]
        if missing:
            return Result("camera", False, f"enabled={enabled} but no frame from {missing}")
        return Result("camera", True, f"enabled={enabled}  all received a frame")
    except Exception as e:
        return Result("camera", False, f"{type(e).__name__}: {e}")


def check_mobile(base_type: str, sim: bool) -> Result:
    if base_type == "legs":
        return Result("mobile", True, "base_type=legs (skipped)")
    try:
        from unitree_lerobot.eval_robot.robot_control.mobile_control import (
            G1_Mobile_Lift_Controller,
        )

        ctrl = G1_Mobile_Lift_Controller(
            base_type=base_type, r3_controller=False, fps=30.0, simulation_mode=sim
        )
        h = float(ctrl.g1_height_state_array_out[0])
        extra = ""
        if base_type == "mobile_lift" and ctrl.g1_move_state_array_out is not None:
            extra = f" odom=({float(ctrl.g1_move_state_array_out[0]):.3f}, {float(ctrl.g1_move_state_array_out[1]):.3f})"
        return Result("mobile", True, f"base={base_type} height={h:.3f}{extra}")
    except Exception as e:
        return Result("mobile", False, f"{type(e).__name__}: {e}")


def run(args: argparse.Namespace) -> int:
    results: list[Result] = []

    if args.arm:
        results.append(check_arm(args.arm, args.sim))
    if args.ee:
        results.append(check_ee(args.ee, args.sim, args.ee_wait))
    if args.image_host:
        results.append(check_camera(args.image_host))
    if args.base_type:
        results.append(check_mobile(args.base_type, args.sim))

    if not results:
        logger_mp.error("Nothing to check — pass at least one of --arm / --ee / --image-host / --base-type")
        return 2

    logger_mp.info("=" * 60)
    logger_mp.info("Component self-check results")
    logger_mp.info("=" * 60)
    all_ok = True
    for r in results:
        tag = "OK  " if r.ok else "FAIL"
        logger_mp.info(f"[{tag}] {r.name:<7} {r.detail}")
        all_ok = all_ok and r.ok
    logger_mp.info("=" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(3)
