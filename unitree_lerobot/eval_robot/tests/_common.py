"""Shared helpers for component smoke tests.

These must NOT import UnitreeRobot — the whole point of the smoke tests is to
exercise the base components (robot_control/, image_server/) on their own.
"""

from __future__ import annotations

import argparse
import threading
import time
from typing import Callable

try:
    from sshkeyboard import listen_keyboard, stop_listening
    _HAS_SSHKB = True
except Exception:  # pragma: no cover - sshkeyboard is an optional runtime dep
    listen_keyboard = None
    stop_listening = None
    _HAS_SSHKB = False


# ----------------------------------------------------------------------
# Keyboard listener
# ----------------------------------------------------------------------

class KeyListener:
    """Thin wrapper around sshkeyboard that runs in a background thread.

    Usage:
        kl = KeyListener(on_press=lambda k: ...)
        with kl:
            while ...:
                ...
    """

    def __init__(self, on_press: Callable[[str], None]):
        if not _HAS_SSHKB:
            raise RuntimeError(
                "sshkeyboard is not installed; install it or run without keyboard controls."
            )
        self._on_press = on_press
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=listen_keyboard,
            kwargs={"on_press": self._on_press, "until": None, "sequential": False},
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        try:
            stop_listening()
        except Exception:
            pass
        self._thread.join(timeout=1.0)
        self._thread = None

    def __enter__(self) -> "KeyListener":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


# ----------------------------------------------------------------------
# Loop pacing + throttled logging
# ----------------------------------------------------------------------

def loop_at(hz: float, started_at: float) -> None:
    """Sleep to hold the given frequency given a loop start timestamp."""
    if hz <= 0:
        return
    elapsed = time.perf_counter() - started_at
    time.sleep(max(0.0, 1.0 / hz - elapsed))


class Throttle:
    """Fires at most every `period` seconds. Returns True when it fires."""

    def __init__(self, period: float):
        self._period = max(float(period), 0.0)
        self._last: float = -float("inf")

    def __call__(self) -> bool:
        now = time.perf_counter()
        if now - self._last >= self._period:
            self._last = now
            return True
        return False


# ----------------------------------------------------------------------
# Confirmation gate
# ----------------------------------------------------------------------

def confirm_or_exit(
    prompt: str = "Press 's' + Enter to continue, anything else to abort: ",
    *,
    assume_yes: bool = False,
) -> None:
    """Blocking stdin confirmation before sending motion commands.

    `assume_yes` skips the prompt (use for scripted / CI runs).
    """
    if assume_yes:
        return
    ans = input(prompt)
    if ans.strip().lower() != "s":
        raise SystemExit("Aborted by user.")


# ----------------------------------------------------------------------
# Shared argparse surface
# ----------------------------------------------------------------------

def add_common_args(p: argparse.ArgumentParser, *, motion: bool = True) -> None:
    """Register flags shared across all component smoke tests.

    If `motion` is False, --duration and --read-only / --yes are omitted.
    """
    p.add_argument("--sim", action="store_true", help="Use DDS domain 1 (simulation)")
    p.add_argument("--hz", type=float, default=50.0, help="Control-loop frequency (Hz)")
    if motion:
        p.add_argument(
            "--duration",
            type=float,
            default=10.0,
            help="Motion duration in seconds; 0 = until 'q' is pressed",
        )
        p.add_argument(
            "--read-only",
            action="store_true",
            help="Only stream state at 1 Hz; never send motion commands",
        )
        p.add_argument(
            "-y",
            "--yes",
            action="store_true",
            dest="assume_yes",
            help="Skip the interactive motion confirmation prompt",
        )


def motion_loop(
    *,
    hz: float,
    duration: float,
    on_tick: Callable[[float, float], None],
    on_state: Callable[[], str] | None = None,
    state_log_period: float = 1.0,
) -> None:
    """Run a motion loop with keyboard 'q' stop + optional throttled state log.

    Arguments:
        hz              : loop frequency
        duration        : seconds; <= 0 means "run until 'q'"
        on_tick(t, dt)  : per-loop callback receiving (absolute_t, elapsed_since_start)
        on_state()      : returns a short status string, logged at `state_log_period`
        state_log_period: seconds between state log lines

    The caller is responsible for printing via its own logger; this helper only
    arranges the loop skeleton and key listener.
    """
    import logging_mp

    logger_mp = logging_mp.getLogger("tests._common.motion_loop")
    stop = {"v": False}

    def _on_press(key: str) -> None:
        if key == "q":
            stop["v"] = True
            logger_mp.info("q pressed, stopping.")

    throttle = Throttle(state_log_period)
    with KeyListener(_on_press):
        start = time.perf_counter()
        while not stop["v"]:
            t = time.perf_counter()
            if duration > 0 and t - start >= duration:
                break
            loop_t0 = t
            on_tick(t, t - start)
            if on_state is not None and throttle():
                logger_mp.info(on_state())
            loop_at(hz, loop_t0)
