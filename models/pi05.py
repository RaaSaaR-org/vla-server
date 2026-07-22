"""
@file pi05.py
@description pi0.5 model backend stub.

This is a placeholder for Peter's GPU server running pi0.5 via LeRobot's
async inference (gRPC policy_server). Full implementation in TASK-078.

For now, this can run in stub mode, producing sine-wave actions for testing.
"""

import logging
import math
import os
import time

from .base import ModelConfig, PredictResult, VLAModel

logger = logging.getLogger(__name__)

# SO-101 default (6 joints); G1 sim rollouts use 29. Override via the
# action_dim constructor param or the VLA_ACTION_DIM env var.
DEFAULT_ACTION_DIM = 6
CHUNK_SIZE = 50

# Sine-wave stub parameters (realistic resting pose). For action dims > 6
# the pattern repeats via j % len(...) with a per-index phase shift.
_OFFSETS = [0.0, -0.5, 1.0, -0.3, 0.0, 0.5]
_AMPLITUDES = [0.20, 0.30, 0.25, 0.15, 0.35, 0.40]
_FREQUENCIES = [0.10, 0.15, 0.20, 0.30, 0.50, 0.70]


def _resolve_action_dim(action_dim: int | None) -> int:
    """Resolve the action dimension.

    Order: explicit param -> VLA_ACTION_DIM env var -> DEFAULT_ACTION_DIM.
    Non-integer or non-positive values fall back to the default with a
    logged warning instead of crashing the server.
    """
    raw: object = action_dim if action_dim is not None else os.environ.get("VLA_ACTION_DIM")
    if raw is None:
        return DEFAULT_ACTION_DIM
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning(
            f"Invalid action_dim {raw!r} (expected positive int); "
            f"falling back to {DEFAULT_ACTION_DIM}"
        )
        return DEFAULT_ACTION_DIM
    if value <= 0:
        logger.warning(
            f"action_dim must be positive, got {value}; "
            f"falling back to {DEFAULT_ACTION_DIM}"
        )
        return DEFAULT_ACTION_DIM
    return value


class Pi05Model(VLAModel):
    """pi0.5 stub — sine-wave actions, no ML dependencies.

    TODO(TASK-078): Replace with real pi0.5 LeRobot policy loading.
    """

    def __init__(
        self,
        model_path: str = "",
        device: str = "cpu",
        action_dim: int | None = None,
    ):
        self.model_path = model_path
        self.device = device
        self._action_dim = _resolve_action_dim(action_dim)
        self._loaded = False
        self._step = 0
        self._active_adapter_id: str | None = None

    def load(self) -> None:
        self._loaded = True
        self._step = 0
        logger.info(
            f"Pi05Model loaded (stub mode — sine-wave actions, "
            f"action_dim={self._action_dim})"
        )

    def predict(
        self,
        images: dict[str, str],
        state: list[float],
        task: str,
    ) -> PredictResult:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        t_start = time.perf_counter()
        actions: list[list[float]] = []

        n = len(_OFFSETS)
        for i in range(CHUNK_SIZE):
            t = (self._step + i) / 30.0
            action = [
                _OFFSETS[j % n] + _AMPLITUDES[j % n] * math.sin(
                    2.0 * math.pi * _FREQUENCIES[j % n] * t + j * 0.5
                )
                for j in range(self._action_dim)
            ]
            actions.append(action)

        self._step += CHUNK_SIZE
        inference_time_ms = (time.perf_counter() - t_start) * 1000
        return PredictResult(actions=actions, inference_time_ms=inference_time_ms)

    def reset(self) -> None:
        self._step = 0

    def info(self) -> ModelConfig:
        return ModelConfig(
            action_dim=self._action_dim,
            chunk_size=CHUNK_SIZE,
            cameras=["front"],
            state_dim=self._action_dim,
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def is_stub(self) -> bool:
        return True  # pi0.5 backend is stub-only until TASK-078

    @property
    def active_adapter_id(self) -> str | None:
        return self._active_adapter_id

    def load_adapter(self, adapter_path: str, adapter_id: str | None = None) -> dict:
        """Stub adapter loader — records the id without loading any weights.

        Useful for end-to-end testing of the /load-adapter wiring without ML deps.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded; cannot apply adapter")
        name = adapter_id or f"adapter-{int(time.time() * 1000)}"
        self._active_adapter_id = name
        logger.info(f"Pi05Model: stub-loaded adapter '{name}' from {adapter_path}")
        return {
            "adapter_id": name,
            "info": {"strategy": "stub", "path": adapter_path},
        }
