"""
@file stub.py
@description Dependency-free stub backend — sine-wave actions, no ML imports.

This is the code that used to live in `models/pi05.py`. It was moved here on
2026-08-02, when pi0.5 got a real LeRobot backend: `--stub` used to be wired to
the pi05 module precisely *because* pi05 was fake, so a real implementation
would have silently taken torch, transformers and a 3B checkpoint along into
every stub run.

Use it for wiring tests, `/predict` contract checks and adapter hot-swap tests
on machines with no GPU and no ML stack. It is selected by `--stub` /
`VLA_STUB=1`, independent of which model name is configured.
"""

import logging
import math
import os
import time

from .base import ModelConfig, PredictResult, VLAModel

logger = logging.getLogger(__name__)

# SO-101 default — deliberately NOT the G1/Dex3 dimensions. The stub must not
# look like a plausible Apple-PnP policy by accident; anything that reaches the
# sim with these numbers is a wiring bug and should be obvious as one.
# 29 (G1) is available, but only when someone asks for it by name: the
# action_dim constructor param, action_dim in config.yaml, or VLA_ACTION_DIM.
DEFAULT_ACTION_DIM = 6
CHUNK_SIZE = 50

# Sine-wave stub parameters (realistic resting pose). Beyond the sixth index
# the pattern repeats via j % len(...), with j * 0.5 keeping the phases apart.
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


class StubModel(VLAModel):
    """Sine-wave actions, no ML dependencies."""

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
            f"StubModel loaded (sine-wave actions, no ML deps, "
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
        return True

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
        logger.info(f"StubModel: stub-loaded adapter '{name}' from {adapter_path}")
        return {
            "adapter_id": name,
            "info": {"strategy": "stub", "path": adapter_path},
        }
