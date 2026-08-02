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
import time

from .base import ModelConfig, PredictResult, VLAModel

logger = logging.getLogger(__name__)

# SO-101 defaults — deliberately NOT the G1/Dex3 dimensions. The stub must not
# look like a plausible Apple-PnP policy; anything that reaches the sim with
# these numbers is a wiring bug and should be obvious as one.
ACTION_DIM = 6
CHUNK_SIZE = 50

# Sine-wave stub parameters (realistic resting pose)
_OFFSETS = [0.0, -0.5, 1.0, -0.3, 0.0, 0.5]
_AMPLITUDES = [0.20, 0.30, 0.25, 0.15, 0.35, 0.40]
_FREQUENCIES = [0.10, 0.15, 0.20, 0.30, 0.50, 0.70]


class StubModel(VLAModel):
    """Sine-wave actions, no ML dependencies."""

    def __init__(self, model_path: str = "", device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self._loaded = False
        self._step = 0
        self._active_adapter_id: str | None = None

    def load(self) -> None:
        self._loaded = True
        self._step = 0
        logger.info("StubModel loaded (sine-wave actions, no ML deps)")

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

        for i in range(CHUNK_SIZE):
            t = (self._step + i) / 30.0
            action = [
                _OFFSETS[j] + _AMPLITUDES[j] * math.sin(
                    2.0 * math.pi * _FREQUENCIES[j] * t + j * 0.5
                )
                for j in range(ACTION_DIM)
            ]
            actions.append(action)

        self._step += CHUNK_SIZE
        inference_time_ms = (time.perf_counter() - t_start) * 1000
        return PredictResult(actions=actions, inference_time_ms=inference_time_ms)

    def reset(self) -> None:
        self._step = 0

    def info(self) -> ModelConfig:
        return ModelConfig(
            action_dim=ACTION_DIM,
            chunk_size=CHUNK_SIZE,
            cameras=["front"],
            state_dim=6,
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
