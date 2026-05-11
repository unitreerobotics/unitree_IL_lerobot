"""Unified Unitree robot abstraction for policy evaluation.

Composes arm + end-effector + camera + mobile base based on device type.
Does NOT modify the underlying components in `image_server/` or `robot_control/`.

Typical usage
-------------
    robot = UnitreeRobot(cfg)
    robot.start()                         # init all subsystems
    obs = robot.get_observation()         # dict with images + observation.state
    robot.execute_action(action_np)       # dispatches to arm / ee / mobile
    robot.reset_to(init_pose, steps=30)   # interpolated reset
    robot.close()

The action/state vectors always follow the layout
    [arm (arm_dof) | left_ee (ee_dof) | right_ee (ee_dof) | mobile (mobile_action_dim)]
"""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from multiprocessing import Array, Lock, Value
from multiprocessing.sharedctypes import SynchronizedArray
from typing import Any, Literal

import cv2
import numpy as np
import torch

from unitree_lerobot.eval_robot.image_server.image_client import ImageClient
from unitree_lerobot.eval_robot.utils.utils import to_list, to_scalar

import logging_mp

logger_mp = logging_mp.getLogger(__name__)
logger_mp.setLevel(logging_mp.INFO)


# ----------------------------------------------------------------------
# Lazy class reference
# ----------------------------------------------------------------------

ClassRef = "type | str | None"  # purely informative; kept as Any at type-level


def _resolve(ref):
    """Accepts either a class or a ``"module.path:ClassName"`` string and returns
    the class. Import happens lazily the first time the spec is used, so the
    module import graph does not pull every controller's dependencies at once.
    """
    if ref is None or not isinstance(ref, str):
        return ref
    module_path, sep, qualname = ref.partition(":")
    if not sep or not qualname:
        raise ValueError(
            f"Invalid class reference '{ref}'. Expected 'module.path:ClassName'."
        )
    obj = importlib.import_module(module_path)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


# ----------------------------------------------------------------------
# Device-type registry
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class ArmSpec:
    """Arm type — dual-arm controller + IK solver.

    `controller` / `ik_solver` may be a class or a lazy ``"module:Name"`` ref.
    """

    controller: Any
    ik_solver: Any
    dof: int

    def build_controller(self, *, motion_mode: bool, simulation_mode: bool):
        return _resolve(self.controller)(motion_mode=motion_mode, simulation_mode=simulation_mode)

    def build_ik(self):
        return _resolve(self.ik_solver)()


@dataclass(frozen=True)
class EESpec:
    """End-effector type — hand/gripper controller + shared-memory shape."""

    controller: Any
    dof: int
    shared_mem_type: Literal["array", "value"]
    shared_mem_size: int = 0         # only used when type == "array"
    out_len: int | None = None

    def build_controller(self, left_in, right_in, lock, state_arr, action_arr, *, simulation_mode: bool):
        return _resolve(self.controller)(
            left_in, right_in, lock, state_arr, action_arr, simulation_mode=simulation_mode
        )


@dataclass(frozen=True)
class BaseSpec:
    """Mobile-base type. ``controller is None`` means no mobile subsystem (``legs``)."""

    controller: Any          # None for legs; lazy ref or class otherwise
    action_dim: int          # 0 / 1 / 3

    def build_controller(self, *, base_type: str, fps: float, simulation_mode: bool):
        if self.controller is None:
            return None
        return _resolve(self.controller)(
            base_type=base_type, r3_controller=False, fps=fps, simulation_mode=simulation_mode
        )


ARM_REGISTRY: dict[str, ArmSpec] = {}
EE_REGISTRY: dict[str, EESpec] = {}
BASE_REGISTRY: dict[str, BaseSpec] = {}


def register_arm(key: str, spec: ArmSpec) -> ArmSpec:
    if key in ARM_REGISTRY:
        raise ValueError(f"arm '{key}' already registered")
    ARM_REGISTRY[key] = spec
    return spec


def register_ee(key: str, spec: EESpec) -> EESpec:
    key = key.lower()
    if key in EE_REGISTRY:
        raise ValueError(f"ee '{key}' already registered")
    EE_REGISTRY[key] = spec
    return spec


def register_base(key: str, spec: BaseSpec) -> BaseSpec:
    if key in BASE_REGISTRY:
        raise ValueError(f"base '{key}' already registered")
    BASE_REGISTRY[key] = spec
    return spec


# ----------------------------------------------------------------------
# Built-in device types
#
# References are strings so each concrete SDK (Dex3, inspire, brainco, mobile
# lift, ...) is only imported the first time its spec is actually used. This
# keeps `import unitree_lerobot.eval_robot.robot` lightweight and lets a
# deployment ship without, e.g., the brainco SDK as long as nobody asks for it.
# ----------------------------------------------------------------------

_ARM_MOD = "unitree_lerobot.eval_robot.robot_control.robot_arm"
_IK_MOD = "unitree_lerobot.eval_robot.robot_control.robot_arm_ik"
_HAND_UNITREE_MOD = "unitree_lerobot.eval_robot.robot_control.robot_hand_unitree"
_HAND_INSPIRE_MOD = "unitree_lerobot.eval_robot.robot_control.robot_hand_inspire"
_HAND_BRAINCO_MOD = "unitree_lerobot.eval_robot.robot_control.robot_hand_brainco"
_MOBILE_MOD = "unitree_lerobot.eval_robot.robot_control.mobile_control"

register_arm("G1_29", ArmSpec(
    controller=f"{_ARM_MOD}:G1_29_ArmController",
    ik_solver=f"{_IK_MOD}:G1_29_ArmIK",
    dof=14,
))
register_arm("G1_23", ArmSpec(
    controller=f"{_ARM_MOD}:G1_23_ArmController",
    ik_solver=f"{_IK_MOD}:G1_23_ArmIK",
    dof=14,
))

register_ee("dex3", EESpec(
    controller=f"{_HAND_UNITREE_MOD}:Dex3_1_Controller",
    dof=7, shared_mem_type="array", shared_mem_size=7,
))
register_ee("dex1", EESpec(
    controller=f"{_HAND_UNITREE_MOD}:Dex1_1_Gripper_Controller",
    dof=1, shared_mem_type="value",
))
register_ee("inspire1", EESpec(
    controller=f"{_HAND_INSPIRE_MOD}:Inspire_Controller",
    dof=6, shared_mem_type="array", shared_mem_size=6,
))
register_ee("brainco", EESpec(
    controller=f"{_HAND_BRAINCO_MOD}:Brainco_Controller",
    dof=6, shared_mem_type="array", shared_mem_size=6,
))

register_base("legs",        BaseSpec(controller=None, action_dim=0))
register_base("only_height", BaseSpec(
    controller=f"{_MOBILE_MOD}:G1_Mobile_Lift_Controller", action_dim=1,
))
register_base("mobile_lift", BaseSpec(
    controller=f"{_MOBILE_MOD}:G1_Mobile_Lift_Controller", action_dim=3,
))


@dataclass(frozen=True)
class ActionLayout:
    """Slice layout of the concatenated [arm | left_ee | right_ee | mobile] vector.

    All arithmetic over action / state vectors should go through this object so
    no caller has to recompute offsets by hand.
    """

    arm_dof: int
    ee_dof: int
    mobile_action_dim: int

    @property
    def dim(self) -> int:
        return self.arm_dof + 2 * self.ee_dof + self.mobile_action_dim

    @property
    def arm(self) -> slice:
        return slice(0, self.arm_dof)

    @property
    def left_ee(self) -> slice:
        return slice(self.arm_dof, self.arm_dof + self.ee_dof)

    @property
    def right_ee(self) -> slice:
        return slice(self.arm_dof + self.ee_dof, self.arm_dof + 2 * self.ee_dof)

    @property
    def mobile(self) -> slice:
        return slice(self.arm_dof + 2 * self.ee_dof, self.dim)

    def validate(self, vec: np.ndarray, *, source: str = "action") -> None:
        if vec.shape[0] != self.dim:
            raise ValueError(
                f"{source}: expected dim {self.dim} "
                f"[arm={self.arm_dof} + 2*ee={self.ee_dof} + mobile={self.mobile_action_dim}], "
                f"got {vec.shape[0]}."
            )


# ----------------------------------------------------------------------
# Helpers (kept local to avoid touching utils.py / make_robot.py callers)
# ----------------------------------------------------------------------

def _bgr_to_tensor_rgb(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# ----------------------------------------------------------------------
# Subsystems
# ----------------------------------------------------------------------

class _EndEffector:
    """Wraps an EE controller + its shared-memory handles behind a simple API."""

    def __init__(self, ee_key: str, simulation_mode: bool):
        spec = EE_REGISTRY[ee_key]
        self.key = ee_key
        self.spec = spec
        self.dof = spec.dof

        mem_type = spec.shared_mem_type.lower()
        out_len = int(spec.out_len) if spec.out_len is not None else 2 * spec.dof

        self._lock = Lock()
        if mem_type == "array":
            left_in = Array("d", spec.shared_mem_size, lock=True)
            right_in = Array("d", spec.shared_mem_size, lock=True)
        elif mem_type == "value":
            left_in = Value("d", 0.0, lock=True)
            right_in = Value("d", 0.0, lock=True)
        else:
            raise ValueError(f"Unknown shared_mem_type '{mem_type}' for EE '{ee_key}'")

        state_arr = Array("d", out_len, lock=False)
        action_arr = Array("d", out_len, lock=False)

        self._left_in = left_in
        self._right_in = right_in
        self._state = state_arr
        self._action = action_arr
        self._controller = spec.build_controller(
            left_in, right_in, self._lock, state_arr, action_arr, simulation_mode=simulation_mode
        )

    # --- state ---
    def read_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (left_state, right_state) as float64 numpy arrays of length `dof`."""
        with self._lock:
            full = np.array(self._state[:], dtype=np.float64)
        return full[: self.dof], full[self.dof : 2 * self.dof]

    # --- action ---
    def send(self, left_action: np.ndarray, right_action: np.ndarray) -> None:
        if isinstance(self._left_in, SynchronizedArray):
            self._left_in[:] = to_list(left_action)
            self._right_in[:] = to_list(right_action)
        elif hasattr(self._left_in, "value"):
            self._left_in.value = to_scalar(left_action)
            self._right_in.value = to_scalar(right_action)

    # --- compat: expose the same dict as old make_robot.setup_robot_interface ---
    @property
    def shared_mem(self) -> dict[str, Any]:
        return {
            "left": self._left_in,
            "right": self._right_in,
            "state": self._state,
            "action": self._action,
            "lock": self._lock,
        }

    @property
    def controller(self):
        return self._controller


class _MobileBase:
    """Wraps the mobile-lift controller; no-op when ``base_type == 'legs'``."""

    def __init__(self, base_type: str, fps: float, simulation_mode: bool):
        if base_type not in BASE_REGISTRY:
            raise ValueError(f"Unknown base_type '{base_type}'. Known: {list(BASE_REGISTRY)}")
        spec = BASE_REGISTRY[base_type]
        self.base_type = base_type
        self.action_dim = spec.action_dim
        self._controller = spec.build_controller(
            base_type=base_type, fps=float(fps), simulation_mode=simulation_mode
        )

    @property
    def controller(self):
        return self._controller

    def read_state(self) -> np.ndarray:
        if self._controller is None or self.base_type == "legs":
            return np.array([], dtype=np.float64)
        height = float(self._controller.g1_height_state_array_out[0])
        if self.base_type == "mobile_lift":
            return np.array(
                [
                    height,
                    float(self._controller.g1_move_state_array_out[0]),
                    float(self._controller.g1_move_state_array_out[1]),
                ],
                dtype=np.float64,
            )
        return np.array([height], dtype=np.float64)

    def send(self, mobile_action: np.ndarray) -> None:
        if self._controller is None or self.action_dim == 0:
            return
        self._controller.g1_height_action_array_in[0] = float(mobile_action[0])
        if self.base_type == "mobile_lift" and self.action_dim >= 3:
            self._controller.g1_move_action_array_in[0] = float(mobile_action[1])
            self._controller.g1_move_action_array_in[1] = float(mobile_action[2])


class _CameraSet:
    """Wraps ImageClient and exposes a single get_images() call producing the
    `observation.images.*` dict expected by the LeRobot policies."""

    def __init__(self, image_host: str, request_bgr: bool = True):
        self._client = ImageClient(host=image_host, request_bgr=request_bgr)
        self._config = self._client.get_cam_config()
        self._last_status: dict[str, bool] = {}

    @property
    def client(self) -> ImageClient:
        return self._client

    @property
    def config(self) -> dict[str, Any]:
        return self._config

    @property
    def last_status(self) -> dict[str, bool]:
        """Per-camera True/False indicating whether the latest get_images() yielded a frame."""
        return dict(self._last_status)

    def enabled_cameras(self) -> dict[str, bool]:
        cfg = self._config
        return {
            "head": bool(cfg.get("head_camera", {}).get("enable_zmq", False)),
            "left_wrist": bool(cfg.get("left_wrist_camera", {}).get("enable_zmq", False)),
            "right_wrist": bool(cfg.get("right_wrist_camera", {}).get("enable_zmq", False)),
        }

    def get_images(self) -> dict[str, torch.Tensor]:
        cfg = self._config
        out: dict[str, torch.Tensor] = {}
        status: dict[str, bool] = {k: False for k, v in self.enabled_cameras().items() if v}
        try:
            if cfg["head_camera"]["enable_zmq"]:
                head = self._client.get_head_frame()
                if head is not None:
                    w = cfg["head_camera"]["image_shape"][1]
                    out["observation.images.cam_left_high"] = _bgr_to_tensor_rgb(head.bgr[:, : w // 2])
                    out["observation.images.cam_right_high"] = _bgr_to_tensor_rgb(head.bgr[:, w // 2 :])
                    status["head"] = True
                else:
                    logger_mp.warning("Head image is None!")
            if cfg["left_wrist_camera"]["enable_zmq"]:
                lw = self._client.get_left_wrist_frame()
                if lw is not None:
                    out["observation.images.cam_left_wrist"] = _bgr_to_tensor_rgb(lw.bgr)
                    status["left_wrist"] = True
                else:
                    logger_mp.warning("left_wrist image is None!")
            if cfg["right_wrist_camera"]["enable_zmq"]:
                rw = self._client.get_right_wrist_frame()
                if rw is not None:
                    out["observation.images.cam_right_wrist"] = _bgr_to_tensor_rgb(rw.bgr)
                    status["right_wrist"] = True
                else:
                    logger_mp.warning("right_wrist image is None!")
        except Exception as e:
            logger_mp.error(f"[CameraSet.get_images] Failed to process images: {e}")
        self._last_status = status
        return out


# ----------------------------------------------------------------------
# Top-level Robot
# ----------------------------------------------------------------------

class UnitreeRobot:
    """Unified robot façade combining arm, end-effector, cameras and mobile base.

    The `cfg` object only needs the following attributes (all other EvalRealConfig
    fields are ignored here):
        arm          : str    one of ARM_REGISTRY (e.g. "G1_29")
        ee           : str    one of EE_REGISTRY or "" to disable
        base_type    : str    "legs" | "only_height" | "mobile_lift"
        image_host   : str
        motion       : bool   passed to the arm controller
        frequency    : float  control loop frequency (used for mobile ctrl init)
        sim          : bool   optional; defaults to False
    """

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self._sim = bool(getattr(cfg, "sim", False))
        self._frequency = float(getattr(cfg, "frequency", 30.0))

        # Arm
        if cfg.arm not in ARM_REGISTRY:
            raise ValueError(f"Unknown arm '{cfg.arm}'. Known: {list(ARM_REGISTRY)}")
        arm_spec = ARM_REGISTRY[cfg.arm]
        self.arm_ik = arm_spec.build_ik()
        self.arm_ctrl = arm_spec.build_controller(
            motion_mode=bool(getattr(cfg, "motion", False)),
            simulation_mode=self._sim,
        )
        self._arm_dof = int(arm_spec.dof)
        self._arm_key = cfg.arm

        # End-effector
        self._ee: _EndEffector | None = None
        ee_key = (getattr(cfg, "ee", "") or "").lower()
        if ee_key:
            if ee_key not in EE_REGISTRY:
                raise ValueError(f"Unknown end-effector '{cfg.ee}'. Known: {list(EE_REGISTRY)}")
            self._ee = _EndEffector(ee_key, simulation_mode=self._sim)

        # Mobile base
        self._mobile = _MobileBase(
            base_type=getattr(cfg, "base_type", "legs"),
            fps=self._frequency,
            simulation_mode=self._sim,
        )

        # Cameras
        self._cameras = _CameraSet(image_host=cfg.image_host, request_bgr=True)

        # Frozen layout — single source of truth for all slice arithmetic below.
        self._layout = ActionLayout(
            arm_dof=self._arm_dof,
            ee_dof=self.ee_dof,
            mobile_action_dim=self.mobile_action_dim,
        )

        logger_mp.info(f"[UnitreeRobot] {self!r}")

    def __repr__(self) -> str:
        ee_key = self._ee.key if self._ee is not None else "none"
        cams = [k for k, v in self._cameras.enabled_cameras().items() if v] or ["none"]
        return (
            f"UnitreeRobot(arm={self._arm_key}[{self._arm_dof}], ee={ee_key}[{self.ee_dof}], "
            f"base={self.base_type}[{self.mobile_action_dim}], cams={'|'.join(cams)}, "
            f"sim={self._sim}, action_dim={self.action_dim})"
        )

    # ------------------------------------------------------------------
    # Shape / layout properties
    # ------------------------------------------------------------------
    @property
    def arm_dof(self) -> int:
        return self._arm_dof

    @property
    def ee_dof(self) -> int:
        return self._ee.dof if self._ee is not None else 0

    @property
    def has_ee(self) -> bool:
        return self._ee is not None

    @property
    def mobile_action_dim(self) -> int:
        return self._mobile.action_dim

    @property
    def base_type(self) -> str:
        return self._mobile.base_type

    @property
    def action_dim(self) -> int:
        return self._layout.dim

    @property
    def state_dim(self) -> int:
        # Same length as an action vector; the mobile slice differs in meaning
        # (state: [height, odom_x, odom_y] vs action: [height, vx, vyaw]).
        return self._layout.dim

    @property
    def layout(self) -> ActionLayout:
        return self._layout

    # ------------------------------------------------------------------
    # Subsystem accessors (for advanced callers & sim code paths)
    # ------------------------------------------------------------------
    @property
    def cameras(self) -> _CameraSet:
        return self._cameras

    @property
    def image_client(self) -> ImageClient:
        return self._cameras.client

    @property
    def image_config(self) -> dict[str, Any]:
        return self._cameras.config

    @property
    def ee(self) -> _EndEffector | None:
        return self._ee

    @property
    def ee_shared_mem(self) -> dict[str, Any]:
        return self._ee.shared_mem if self._ee is not None else {}

    @property
    def mobile_ctrl(self):
        return self._mobile.controller

    # ------------------------------------------------------------------
    # Observation / state
    # ------------------------------------------------------------------
    def get_arm_state(self) -> np.ndarray:
        q = self.arm_ctrl.get_current_dual_arm_q()
        return np.asarray(q, dtype=np.float64)

    def get_ee_state(self) -> tuple[np.ndarray, np.ndarray]:
        if self._ee is None:
            empty = np.array([], dtype=np.float64)
            return empty, empty
        return self._ee.read_state()

    def get_mobile_state(self) -> np.ndarray:
        return self._mobile.read_state()

    def get_state(self) -> np.ndarray:
        """Full state vector [arm | left_ee | right_ee | mobile]."""
        arm_q = self.get_arm_state()
        left_ee, right_ee = self.get_ee_state()
        mobile = self.get_mobile_state()
        return np.concatenate((arm_q, left_ee, right_ee, mobile), axis=0)

    def get_observation(self, task: str | None = None) -> dict[str, Any]:
        """Policy-ready observation: images + `observation.state` tensor."""
        obs: dict[str, Any] = self._cameras.get_images()
        state = self.get_state()
        obs["observation.state"] = torch.from_numpy(state).float()
        if task is not None:
            obs["task"] = task
        return obs

    def get_observation_status(self) -> dict[str, bool]:
        """Per-subsystem freshness indicator for the most recent get_observation()."""
        status = dict(self._cameras.last_status)
        status["arm"] = True
        status["ee"] = self.has_ee
        status["mobile"] = self.mobile_action_dim > 0
        return status

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------
    def _split_action(
        self, action_np: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        a = np.asarray(action_np, dtype=np.float64)
        self._layout.validate(a, source="execute_action")
        L = self._layout
        return a[L.arm], a[L.left_ee], a[L.right_ee], a[L.mobile]

    def send_arm(self, arm_q: np.ndarray) -> None:
        arm_q = np.asarray(arm_q, dtype=np.float64)
        if arm_q.shape[0] != self._arm_dof:
            raise ValueError(f"send_arm: expected arm_dof {self._arm_dof}, got {arm_q.shape[0]}")
        tau = self.arm_ik.solve_tau(arm_q)
        self.arm_ctrl.ctrl_dual_arm(arm_q, tau)

    def send_ee(
        self,
        left: np.ndarray,
        right: np.ndarray,
        *,
        gate_on_nonzero: bool = False,
    ) -> None:
        """Write left/right EE commands to shared memory.

        `gate_on_nonzero=True` skips the write when both sides are all zero
        (preserves existing finger state when a policy is not actively
        driving the EE).
        """
        if self._ee is None or self.ee_dof == 0:
            return
        if gate_on_nonzero and not (np.any(left) or np.any(right)):
            return
        self._ee.send(left, right)

    def send_mobile(self, mobile_action: np.ndarray) -> None:
        if self.mobile_action_dim == 0:
            return
        mobile_action = np.asarray(mobile_action, dtype=np.float64)
        if mobile_action.shape[0] != self.mobile_action_dim:
            raise ValueError(
                f"send_mobile: expected dim {self.mobile_action_dim}, got {mobile_action.shape[0]}"
            )
        self._mobile.send(mobile_action)

    def execute_action(self, action_np: np.ndarray, *, gate_ee_on_nonzero: bool = True) -> None:
        """Split and dispatch an action vector to all subsystems.

        Expected layout: [arm | left_ee | right_ee | mobile]. The per-subsystem
        `send_*` methods are already no-ops when the subsystem is absent, so no
        outer guards are needed here.
        """
        arm, left_ee, right_ee, mobile = self._split_action(action_np)
        self.send_arm(arm)
        self.send_ee(left_ee, right_ee, gate_on_nonzero=gate_ee_on_nonzero)
        self.send_mobile(mobile)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def current_as_action(self) -> np.ndarray:
        """Return a hold-in-place *action* vector built from the current state.

        Arm and EE slices are identical to `get_state()`. The mobile slice is
        `[height, 0.0, 0.0]` (not `[height, odom_x, odom_y]`) so the returned
        vector is safe to feed back into `execute_action` or to interpolate
        toward a target action pose.
        """
        arm_q = self.get_arm_state()
        left_ee, right_ee = self.get_ee_state()
        if self.mobile_action_dim == 0:
            mobile = np.array([], dtype=np.float64)
        else:
            state = self.get_mobile_state()
            mobile = np.zeros(self.mobile_action_dim, dtype=np.float64)
            if state.size > 0:
                mobile[0] = float(state[0])  # keep current height; velocities stay 0
        return np.concatenate((arm_q, left_ee, right_ee, mobile), axis=0)

    def reset_to(self, target_pose: np.ndarray, steps: int = 30, dt: float | None = None) -> None:
        """Linearly interpolate from the current pose to `target_pose`.

        The start of the interpolation is `current_as_action()` (not raw
        state), so mobile velocity components stay at zero for the duration
        of the reset instead of trying to chase odometry values.

        Uses monotonic-clock pacing so per-loop execution time doesn't
        accumulate into the overall schedule.
        """
        target = np.asarray(target_pose, dtype=np.float64)
        self._layout.validate(target, source="reset_to")
        if not np.all(np.isfinite(target)):
            raise ValueError("reset_to: target_pose contains non-finite values")

        current = self.current_as_action()
        steps = max(int(steps), 1)
        step_dt = dt if dt is not None else 1.0 / max(self._frequency, 1e-3)
        max_delta = float(np.max(np.abs(target - current))) if target.size else 0.0
        logger_mp.info(
            f"[UnitreeRobot] reset_to: {steps} step(s) @ {1.0 / step_dt:.1f} Hz, "
            f"max|Δ|={max_delta:.4f}"
        )

        deadline = time.perf_counter()
        for q in np.linspace(current, target, steps):
            self.execute_action(q, gate_ee_on_nonzero=False)
            deadline += step_dt
            remaining = deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)

    def go_home(self) -> None:
        try:
            self.arm_ctrl.ctrl_dual_arm_go_home()
        except Exception as e:
            logger_mp.error(f"go_home: ctrl_dual_arm_go_home failed: {e}")

    def close(self) -> None:
        # Underlying controllers manage their own daemon threads; nothing to
        # tear down here today. Placeholder for future cleanup.
        pass

    # ------------------------------------------------------------------
    # Context manager sugar
    # ------------------------------------------------------------------
    def __enter__(self) -> "UnitreeRobot":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.go_home()
        finally:
            self.close()
