"""Mobile base / lift smoke test: instantiate G1_Mobile_Lift_Controller, read state, nudge height/velocity.

python -m unitree_lerobot.eval_robot.tests.test_mobile --base-type only_height --height-delta 0.02
python -m unitree_lerobot.eval_robot.tests.test_mobile --base-type mobile_lift --vx 0.05 --vyaw 0.0 --duration 3
"""

from __future__ import annotations

import argparse
import time

import logging_mp

from unitree_lerobot.eval_robot.robot_control.mobile_control import G1_Mobile_Lift_Controller
from unitree_lerobot.eval_robot.tests._common import (
    add_common_args,
    confirm_or_exit,
    motion_loop,
)

logger_mp = logging_mp.getLogger(__name__)
logger_mp.setLevel(logging_mp.INFO)


BASE_TYPES = ("only_height", "mobile_lift")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-type", choices=BASE_TYPES, required=True)
    p.add_argument("--fps", type=float, default=30.0, help="Mobile controller internal fps")
    p.add_argument(
        "--height-delta",
        type=float,
        default=0.02,
        help="Height delta (m) applied on top of the initial height during the test.",
    )
    p.add_argument("--vx", type=float, default=0.0, help="Forward velocity command (mobile_lift only)")
    p.add_argument("--vyaw", type=float, default=0.0, help="Yaw velocity command (mobile_lift only)")
    add_common_args(p)
    p.set_defaults(hz=20.0, duration=3.0)
    return p.parse_args()


def _state_str(ctrl: G1_Mobile_Lift_Controller, base_type: str) -> str:
    h = float(ctrl.g1_height_state_array_out[0])
    if base_type == "mobile_lift" and ctrl.g1_move_state_array_out is not None:
        ox = float(ctrl.g1_move_state_array_out[0])
        oy = float(ctrl.g1_move_state_array_out[1])
        return f"height={h:.3f}  odom=({ox:.3f}, {oy:.3f})"
    return f"height={h:.3f}"


def _stream_readonly(ctrl, base_type: str) -> None:
    logger_mp.info("[test_mobile] --read-only; streaming state at 1 Hz (Ctrl+C to stop).")
    try:
        while True:
            logger_mp.info(_state_str(ctrl, base_type))
            time.sleep(1.0)
    except KeyboardInterrupt:
        return


def run(args: argparse.Namespace) -> None:
    ctrl = G1_Mobile_Lift_Controller(
        base_type=args.base_type,
        r3_controller=False,
        fps=float(args.fps),
        simulation_mode=args.sim,
    )
    logger_mp.info(f"[test_mobile] base_type={args.base_type} initial {_state_str(ctrl, args.base_type)}")

    if args.read_only:
        _stream_readonly(ctrl, args.base_type)
        return

    h0 = float(ctrl.g1_height_state_array_out[0])
    h_target = h0 + float(args.height_delta)
    logger_mp.info(
        f"[test_mobile] Will hold height={h_target:.3f} (delta {args.height_delta:+.3f}) "
        + (f"and command vx={args.vx} vyaw={args.vyaw}" if args.base_type == "mobile_lift" else "")
    )
    confirm_or_exit(assume_yes=args.assume_yes)

    def tick(_t: float, _elapsed: float) -> None:
        ctrl.g1_height_action_array_in[0] = float(h_target)
        if args.base_type == "mobile_lift":
            ctrl.g1_move_action_array_in[0] = float(args.vx)
            ctrl.g1_move_action_array_in[1] = float(args.vyaw)

    def state() -> str:
        return f"[test_mobile] {_state_str(ctrl, args.base_type)}"

    try:
        motion_loop(hz=args.hz, duration=args.duration, on_tick=tick, on_state=state)
    finally:
        logger_mp.info("[test_mobile] Zeroing motion command and restoring initial height.")
        try:
            ctrl.g1_height_action_array_in[0] = h0
            if args.base_type == "mobile_lift":
                ctrl.g1_move_action_array_in[0] = 0.0
                ctrl.g1_move_action_array_in[1] = 0.0
        except Exception as e:
            logger_mp.error(f"[test_mobile] reset failed: {e}")
        logger_mp.info(f"[test_mobile] final {_state_str(ctrl, args.base_type)}")


if __name__ == "__main__":
    run(parse_args())
