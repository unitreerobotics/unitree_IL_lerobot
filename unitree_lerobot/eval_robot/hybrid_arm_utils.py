"""Pure helpers for dataset-vision, real-arm hybrid inference."""

from __future__ import annotations

import numpy as np


ARM_DOF = 14
FULL_STATE_DOF = 28


def compose_hybrid_state(dataset_state, real_arm_state) -> np.ndarray:
    """Replace the dataset arm state with the current real G1 arm state."""
    dataset = np.asarray(dataset_state, dtype=np.float32).reshape(-1)
    real_arm = np.asarray(real_arm_state, dtype=np.float32).reshape(-1)

    if dataset.shape != (FULL_STATE_DOF,):
        raise ValueError(f"Expected dataset state shape ({FULL_STATE_DOF},), got {dataset.shape}")
    if real_arm.shape != (ARM_DOF,):
        raise ValueError(f"Expected real arm state shape ({ARM_DOF},), got {real_arm.shape}")
    if not np.all(np.isfinite(dataset)):
        raise ValueError("Dataset state contains NaN or Inf")
    if not np.all(np.isfinite(real_arm)):
        raise ValueError("Real arm state contains NaN or Inf")

    hybrid = dataset.copy()
    hybrid[:ARM_DOF] = real_arm
    return hybrid


def extract_arm_chunk(action_chunk) -> np.ndarray:
    """Validate a full ACT chunk and return its first 14 action dimensions."""
    actions = np.asarray(action_chunk, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != FULL_STATE_DOF:
        raise ValueError(f"Expected action chunk shape (T, {FULL_STATE_DOF}), got {actions.shape}")
    if actions.shape[0] == 0:
        raise ValueError("Action chunk is empty")
    if not np.all(np.isfinite(actions)):
        raise ValueError("Action chunk contains NaN or Inf")
    return actions[:, :ARM_DOF].copy()


def limit_arm_target(predicted_target, current_arm_state, max_delta_rad: float) -> np.ndarray:
    """Clamp each target joint to a maximum delta from the latest real state."""
    predicted = np.asarray(predicted_target, dtype=np.float32).reshape(-1)
    current = np.asarray(current_arm_state, dtype=np.float32).reshape(-1)

    if predicted.shape != (ARM_DOF,) or current.shape != (ARM_DOF,):
        raise ValueError(f"Expected two ({ARM_DOF},) vectors, got {predicted.shape} and {current.shape}")
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(current)):
        raise ValueError("Arm target or state contains NaN or Inf")
    if not np.isfinite(max_delta_rad) or max_delta_rad <= 0:
        raise ValueError("max_delta_rad must be finite and greater than zero")

    delta = np.clip(predicted - current, -max_delta_rad, max_delta_rad)
    return current + delta


def chunk_timestep_range(
    anchor_timestep: int,
    chunk_length: int,
    actions_per_inference: int,
    current_timestep: int,
) -> range:
    """Return the non-stale absolute timesteps supplied by an action chunk."""
    if anchor_timestep < 0 or current_timestep < 0:
        raise ValueError("Timesteps must be non-negative")
    if chunk_length <= 0 or actions_per_inference <= 0:
        raise ValueError("Chunk lengths must be greater than zero")

    stop = anchor_timestep + min(chunk_length, actions_per_inference)
    return range(max(anchor_timestep, current_timestep), stop)
