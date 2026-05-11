"""End-effector smoke test: instantiate an EE controller in isolation, read state, send open/close.

python -m unitree_lerobot.eval_robot.tests.test_ee --ee dex3 --amplitude 0.15 --period 3.0
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from multiprocessing import Array, Lock, Value

import numpy as np

import logging_mp

from unitree_lerobot.eval_robot.robot_control.robot_hand_brainco import Brainco_Controller
from unitree_lerobot.eval_robot.robot_control.robot_hand_inspire import Inspire_Controller
from unitree_lerobot.eval_robot.robot_control.robot_hand_unitree import (
    Dex1_1_Gripper_Controller,
    Dex3_1_Controller,
)
from unitree_lerobot.eval_robot.tests._common import (
    add_common_args,
    confirm_or_exit,
    motion_loop,
)

logger_mp = logging_mp.getLogger(__name__)
logger_mp.setLevel(logging_mp.INFO)


@dataclass(frozen=True)
class EESpec:
    ctrl: type
    dof: int
    shared_mem_type: str  # "array" or "value"
    shared_mem_size: int = 0
    out_len: int | None = None


EES: dict[str, EESpec] = {
    "dex3":     EESpec(Dex3_1_Controller,         dof=7, shared_mem_type="array", shared_mem_size=7),
    "dex1":     EESpec(Dex1_1_Gripper_Controller, dof=1, shared_mem_type="value"),
    "inspire1": EESpec(Inspire_Controller,        dof=6, shared_mem_type="array", shared_mem_size=6),
    "brainco":  EESpec(Brainco_Controller,        dof=6, shared_mem_type="array", shared_mem_size=6),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ee", choices=sorted(EES), required=True)
    p.add_argument(
        "--amplitude",
        type=float,
        default=0.15,
        help="Additive delta applied symmetrically to all fingers; stay small.",
    )
    p.add_argument("--period", type=float, default=3.0, help="Full open-close cycle period (s)")
    add_common_args(p)
    p.set_defaults(hz=50.0)
    return p.parse_args()


def _build_shared_mem(spec: EESpec):
    lock = Lock()
    if spec.shared_mem_type.lower() == "array":
        left_in = Array("d", spec.shared_mem_size, lock=True)
        right_in = Array("d", spec.shared_mem_size, lock=True)
    else:
        left_in = Value("d", 0.0, lock=True)
        right_in = Value("d", 0.0, lock=True)
    out_len = int(spec.out_len) if spec.out_len is not None else 2 * spec.dof
    state_arr = Array("d", out_len, lock=False)
    action_arr = Array("d", out_len, lock=False)
    return left_in, right_in, lock, state_arr, action_arr


def _read_state(spec: EESpec, lock, state_arr) -> tuple[np.ndarray, np.ndarray]:
    with lock:
        full = np.array(state_arr[:], dtype=np.float64)
    return full[: spec.dof], full[spec.dof : 2 * spec.dof]


def _write_action(spec: EESpec, left_in, right_in, target_value: float) -> None:
    if spec.shared_mem_type.lower() == "array":
        vec = [float(target_value)] * spec.shared_mem_size
        left_in[:] = vec
        right_in[:] = vec
    else:
        left_in.value = float(target_value)
        right_in.value = float(target_value)


def _stream_readonly(spec: EESpec, lock, state_arr) -> None:
    logger_mp.info("[test_ee] --read-only; streaming state at 1 Hz (Ctrl+C to stop).")
    try:
        while True:
            left, right = _read_state(spec, lock, state_arr)
            logger_mp.info(f"left={left.tolist()}  right={right.tolist()}")
            time.sleep(1.0)
    except KeyboardInterrupt:
        return


def run(args: argparse.Namespace) -> None:
    spec = EES[args.ee]
    left_in, right_in, lock, state_arr, action_arr = _build_shared_mem(spec)
    ctrl = spec.ctrl(left_in, right_in, lock, state_arr, action_arr, simulation_mode=args.sim)
    logger_mp.info(f"[test_ee] ee={args.ee} dof={spec.dof} mem_type={spec.shared_mem_type}")

    # Give the child control process a moment to flush initial state.
    time.sleep(0.5)

    left, right = _read_state(spec, lock, state_arr)
    logger_mp.info(f"[test_ee] initial state  left={left.tolist()}  right={right.tolist()}")

    if args.read_only:
        _stream_readonly(spec, lock, state_arr)
        return

    logger_mp.info(
        f"[test_ee] Will oscillate all fingers between 0 and {args.amplitude} at period {args.period}s."
    )
    confirm_or_exit(assume_yes=args.assume_yes)

    def tick(_t: float, elapsed: float) -> None:
        phase = 0.5 - 0.5 * np.cos(2.0 * np.pi * elapsed / max(args.period, 1e-3))
        _write_action(spec, left_in, right_in, float(args.amplitude) * phase)

    def state() -> str:
        left, right = _read_state(spec, lock, state_arr)
        return f"[test_ee] left={np.round(left, 3).tolist()} right={np.round(right, 3).tolist()}"

    try:
        motion_loop(hz=args.hz, duration=args.duration, on_tick=tick, on_state=state)
    finally:
        logger_mp.info("[test_ee] Returning to zero command.")
        try:
            _write_action(spec, left_in, right_in, 0.0)
        except Exception as e:
            logger_mp.error(f"[test_ee] reset failed: {e}")


if __name__ == "__main__":
    run(parse_args())
