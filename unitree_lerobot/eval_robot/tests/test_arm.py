"""Arm smoke test: instantiate G1_29/23 arm controller + IK, read q, optional small sine.

python -m unitree_lerobot.eval_robot.tests.test_arm --arm G1_29 --joint 0 --amplitude 0.05 --period 4.0
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import logging_mp

from unitree_lerobot.eval_robot.robot_control.robot_arm import (
    G1_23_ArmController,
    G1_29_ArmController,
)
from unitree_lerobot.eval_robot.robot_control.robot_arm_ik import G1_23_ArmIK, G1_29_ArmIK
from unitree_lerobot.eval_robot.tests._common import (
    add_common_args,
    confirm_or_exit,
    motion_loop,
)

logger_mp = logging_mp.getLogger(__name__)
logger_mp.setLevel(logging_mp.INFO)


ARMS = {
    "G1_29": (G1_29_ArmController, G1_29_ArmIK, 14),
    "G1_23": (G1_23_ArmController, G1_23_ArmIK, 14),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=sorted(ARMS), default="G1_29")
    p.add_argument("--motion", action="store_true", help="Use rt/arm_sdk (motion_mode) instead of rt/lowcmd")
    p.add_argument("--joint", type=int, default=0, help="Joint index to perturb (0..arm_dof-1)")
    p.add_argument("--amplitude", type=float, default=0.05, help="Sine amplitude (rad)")
    p.add_argument("--period", type=float, default=4.0, help="Sine period (s)")
    add_common_args(p)
    p.set_defaults(hz=100.0)
    return p.parse_args()


def _stream_readonly(ctrl) -> None:
    logger_mp.info("[test_arm] --read-only; streaming q at 1 Hz (Ctrl+C to stop).")
    try:
        while True:
            q = np.asarray(ctrl.get_current_dual_arm_q(), dtype=np.float64)
            logger_mp.info(f"q = {q.tolist()}")
            time.sleep(1.0)
    except KeyboardInterrupt:
        return


def run(args: argparse.Namespace) -> None:
    Ctrl, IK, dof = ARMS[args.arm]
    ik = IK()
    ctrl = Ctrl(motion_mode=args.motion, simulation_mode=args.sim)

    q0 = np.asarray(ctrl.get_current_dual_arm_q(), dtype=np.float64)
    logger_mp.info(f"[test_arm] arm={args.arm} dof={dof} current q = {q0.tolist()}")

    if args.read_only:
        _stream_readonly(ctrl)
        return

    joint_idx = int(np.clip(args.joint, 0, dof - 1))
    logger_mp.info(
        f"[test_arm] Will apply sine on joint {joint_idx}: amplitude={args.amplitude} rad, period={args.period}s."
    )
    confirm_or_exit(assume_yes=args.assume_yes)

    def tick(_t: float, elapsed: float) -> None:
        target = q0.copy()
        target[joint_idx] = q0[joint_idx] + args.amplitude * np.sin(
            2.0 * np.pi * elapsed / max(args.period, 1e-3)
        )
        tau = ik.solve_tau(target)
        ctrl.ctrl_dual_arm(target, tau)

    def state() -> str:
        q = np.asarray(ctrl.get_current_dual_arm_q(), dtype=np.float64)
        return f"[test_arm] q[{joint_idx}]={q[joint_idx]:+.4f} (q0={q0[joint_idx]:+.4f})"

    try:
        motion_loop(hz=args.hz, duration=args.duration, on_tick=tick, on_state=state)
    finally:
        logger_mp.info("[test_arm] Returning home.")
        try:
            ctrl.ctrl_dual_arm_go_home()
        except Exception as e:
            logger_mp.error(f"[test_arm] go_home failed: {e}")


if __name__ == "__main__":
    run(parse_args())
