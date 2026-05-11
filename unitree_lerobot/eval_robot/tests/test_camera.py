"""Camera smoke test: instantiate ImageClient, read frames, print/save.

python -m unitree_lerobot.eval_robot.tests.test_camera --image-host 192.168.123.164 --frames 30
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

import logging_mp

from unitree_lerobot.eval_robot.image_server.image_client import ImageClient

logger_mp = logging_mp.getLogger(__name__)
logger_mp.setLevel(logging_mp.INFO)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image-host", default="192.168.123.164")
    p.add_argument("--frames", type=int, default=30, help="Number of sample rounds (per-camera frames attempted)")
    p.add_argument("--timeout", type=float, default=2.0, help="First-frame read timeout (s) per camera")
    p.add_argument("--save-dir", default=None, help="If set, save one representative frame per camera here")
    p.add_argument("--show", action="store_true", help="Display frames via cv2.imshow (needs a display)")
    return p.parse_args()


def _summarize(name: str, frame) -> bool:
    if frame is None or getattr(frame, "bgr", None) is None:
        logger_mp.warning(f"[test_camera] {name}: NO FRAME")
        return False
    img = frame.bgr
    logger_mp.info(f"[test_camera] {name}: shape={img.shape} dtype={img.dtype} mean={float(img.mean()):.1f}")
    return True


def _poll_until(fn, deadline: float):
    while time.perf_counter() < deadline:
        val = fn()
        if val is not None and getattr(val, "bgr", None) is not None:
            return val
        time.sleep(0.02)
    return None


def run(args: argparse.Namespace) -> None:
    client = ImageClient(host=args.image_host, request_bgr=True)
    cam_cfg = client.get_cam_config()

    enabled = {
        "head": cam_cfg.get("head_camera", {}).get("enable_zmq", False),
        "left_wrist": cam_cfg.get("left_wrist_camera", {}).get("enable_zmq", False),
        "right_wrist": cam_cfg.get("right_wrist_camera", {}).get("enable_zmq", False),
    }
    logger_mp.info(f"[test_camera] host={args.image_host} cam enable = {enabled}")

    getters = {
        "head": client.get_head_frame,
        "left_wrist": client.get_left_wrist_frame,
        "right_wrist": client.get_right_wrist_frame,
    }

    # First-pass: make sure each enabled cam yields at least one frame.
    first_ok: dict[str, bool] = {}
    for name, ok in enabled.items():
        if not ok:
            continue
        frame = _poll_until(getters[name], time.perf_counter() + args.timeout)
        first_ok[name] = _summarize(name + " (first)", frame)

    unreachable = [n for n, ok in first_ok.items() if not ok]
    if unreachable:
        logger_mp.error(f"[test_camera] cameras never produced a frame: {unreachable}")

    # Second-pass: sample frames and compute observed FPS + miss rate.
    save_dir = Path(args.save_dir).expanduser().resolve() if args.save_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    active = [n for n, v in enabled.items() if v]
    hits = {k: 0 for k in active}
    misses = {k: 0 for k in active}
    saved: set[str] = set()

    t_start = time.perf_counter()
    for _ in range(args.frames):
        for name in active:
            frame = getters[name]()
            if frame is None or frame.bgr is None:
                misses[name] += 1
                continue
            hits[name] += 1
            if save_dir is not None and name not in saved:
                out = save_dir / f"{name}.png"
                cv2.imwrite(str(out), frame.bgr)
                logger_mp.info(f"[test_camera] saved {out}")
                saved.add(name)
            if args.show:
                cv2.imshow(name, frame.bgr)
        if args.show and cv2.waitKey(1) & 0xFF == ord("q"):
            break
        time.sleep(0.01)
    t_elapsed = time.perf_counter() - t_start

    for name in active:
        c, m = hits[name], misses[name]
        total = c + m
        fps = c / t_elapsed if t_elapsed > 0 else 0.0
        miss_pct = 100.0 * m / total if total > 0 else 0.0
        logger_mp.info(
            f"[test_camera] {name}: {c} frames / {total} polls in {t_elapsed:.2f}s "
            f"(~{fps:.1f} fps, miss {miss_pct:.1f}%)"
        )

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run(parse_args())
