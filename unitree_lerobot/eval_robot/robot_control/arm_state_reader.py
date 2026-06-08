"""Read-only DDS subscriber for the 14 G1-29 arm joints."""

from __future__ import annotations

import threading
import time

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_


G1_29_ARM_INDICES = tuple(range(15, 29))
LOW_STATE_TOPIC = "rt/lowstate"


class G1ArmStateReader:
    """Subscribe to G1 low state without creating a command publisher."""

    def __init__(self, network_interface: str | None = None, timeout_s: float = 10.0):
        ChannelFactoryInitialize(0, network_interface)
        self._subscriber = ChannelSubscriber(LOW_STATE_TOPIC, LowState_)
        self._subscriber.Init()
        self._lock = threading.Lock()
        self._state: np.ndarray | None = None
        self._last_update_s: float | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._read_loop, name="g1-arm-state-reader", daemon=True)
        self._thread.start()
        self.wait_until_ready(timeout_s)

    def _read_loop(self) -> None:
        while not self._closed:
            msg = self._subscriber.Read()
            if msg is None:
                time.sleep(0.002)
                continue
            state = np.asarray([msg.motor_state[index].q for index in G1_29_ARM_INDICES], dtype=np.float32)
            with self._lock:
                self._state = state
                self._last_update_s = time.monotonic()

    def wait_until_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if self._state is not None:
                    return
            time.sleep(0.02)
        self.close()
        raise TimeoutError(f"No G1 low-state message received on {LOW_STATE_TOPIC} within {timeout_s:.1f}s")

    def get_current_dual_arm_q(self, max_age_s: float = 0.25) -> np.ndarray:
        with self._lock:
            state = None if self._state is None else self._state.copy()
            last_update_s = self._last_update_s
        if state is None or last_update_s is None:
            raise RuntimeError("G1 arm state is not available")
        age_s = time.monotonic() - last_update_s
        if age_s > max_age_s:
            raise TimeoutError(f"G1 arm state is stale ({age_s:.3f}s > {max_age_s:.3f}s)")
        return state

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._subscriber.Close()
