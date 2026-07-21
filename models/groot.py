"""
@file groot.py
@description GR00T N1.7 model backend for vla-server.

Connects to the Isaac-GR00T PolicyServer via ZMQ (default port 5555).

Wire protocol (Isaac-GR00T gr00t/policy/server_client.py):
- Request:  msgpack {"endpoint": str, "data": dict, "api_token": str?}
- Response: msgpack; "get_action" returns (action_dict, info_dict),
  errors come back as {"error": str}
- Arrays travel as msgpack_numpy (the server refuses pickle payloads)

Observation format (N1.7, batch B=1, time T=1):
    {"video":    {<camera>: (1, 1, H, W, 3) uint8},   # one entry per video_keys
     "state":    {<name>:   (1, 1, D) float32},
     "language": {<language_key>: [[str]]}}           # language_key default "task"
Action response: {<name>: (B, T, D) float32} — keys concatenated in
action_keys order into (horizon, action_dim) rows.

Stub mode: returns sine-wave actions without any ZMQ or ML deps.
"""

import base64
import io
import logging
import math
import time

import numpy as np
from PIL import Image

from .base import ModelConfig, PredictResult, VLAModel

logger = logging.getLogger(__name__)

# SO-101 defaults
ACTION_DIM = 6  # 5 arm joints + 1 gripper
CHUNK_SIZE = 16  # GR00T action horizon
STATE_DIM = 6
IMAGE_SIZE = 224

# Modality keys must match the embodiment config the checkpoint was
# fine-tuned with (see Isaac-GR00T --embodiment-tag).
DEFAULT_STATE_KEYS = {"single_arm": 5, "gripper": 1}
DEFAULT_ACTION_KEYS = {"single_arm": 5, "gripper": 1}


class GR00TModel(VLAModel):
    """GR00T N1.7 policy via ZMQ connection to an Isaac-GR00T PolicyServer."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        stub: bool = False,
        api_token: str | None = None,
        video_key: str = "front",
        state_keys: dict[str, int] | None = None,
        action_keys: dict[str, int] | None = None,
        timeout_ms: int = 15000,
        ping_retries: int = 3,
        video_keys: list[str] | None = None,
        language_key: str = "task",
        image_size: int | None = IMAGE_SIZE,
    ):
        self.host = host
        self.port = port
        self.stub = stub
        self.api_token = api_token
        # video_keys (multi-camera) wins over the legacy single video_key;
        # video_key stays as a back-compat alias for the first camera.
        self.video_keys = list(video_keys) if video_keys else [video_key]
        self.video_key = self.video_keys[0]
        # Language key inside the observation. Checkpoints fine-tuned with a
        # custom modality config (e.g. g1_dex3) expect
        # "annotation.human.task_description" — the GR00T server asserts on
        # a mismatch in strict mode.
        self.language_key = language_key
        # Square resize edge for camera frames; None sends native resolution.
        self.image_size = image_size
        self.state_keys = dict(state_keys) if state_keys else dict(DEFAULT_STATE_KEYS)
        self.action_keys = dict(action_keys) if action_keys else dict(DEFAULT_ACTION_KEYS)
        self.timeout_ms = timeout_ms
        self.ping_retries = max(1, ping_retries)
        self._socket = None
        self._zmq_ctx = None
        self._loaded = False
        self._step = 0
        self._warned_partial_state = False

    @property
    def _state_dim(self) -> int:
        return sum(self.state_keys.values())

    @property
    def _action_dim(self) -> int:
        return sum(self.action_keys.values())

    def load(self) -> None:
        if self._loaded:
            logger.warning("GR00TModel already loaded, skipping")
            return

        if self.stub:
            logger.info("GR00TModel loaded (stub mode — sine-wave actions)")
            self._loaded = True
            return

        try:
            import zmq  # noqa: F401
        except ImportError:
            raise ImportError(
                "GR00T backend requires the groot extras: uv pip install -e '.[groot]'"
            )

        self._connect()

        # Verify the PolicyServer is reachable. Transport errors are retried
        # (GPU side may boot slower than us); application errors like a bad
        # api_token fail immediately since retrying cannot fix them.
        pong = None
        last_err: Exception | None = None
        for attempt in range(1, self.ping_retries + 1):
            try:
                pong = self._request("ping")
                break
            except RuntimeError as e:
                if "not responding" not in str(e):
                    self.close()
                    raise  # application-level error (e.g. bad token)
                last_err = e
                if attempt < self.ping_retries:
                    logger.warning(
                        f"GR00T ping {attempt}/{self.ping_retries} failed: {e}; "
                        f"retrying in 2s"
                    )
                    time.sleep(2.0)
        if pong is None:
            self.close()
            raise RuntimeError(
                f"GR00T PolicyServer at {self.host}:{self.port} unreachable "
                f"after {self.ping_retries} attempts: {last_err}"
            )
        logger.info(f"GR00T PolicyServer ping: {pong}")

        self._loaded = True
        logger.info(
            f"GR00TModel loaded: ZMQ -> tcp://{self.host}:{self.port} "
            f"(auth={'on' if self.api_token else 'off'})"
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

        if self.stub:
            actions = self._stub_predict(state)
        else:
            actions = self._zmq_predict(images, state, task)

        inference_time_ms = (time.perf_counter() - t_start) * 1000
        return PredictResult(actions=actions, inference_time_ms=inference_time_ms)

    def reset(self) -> None:
        self._step = 0
        if self._socket is not None and not self.stub:
            try:
                self._request("reset", {"options": {}})
            except Exception as e:
                logger.warning(f"GR00T reset failed (best-effort): {e}")

    def info(self) -> ModelConfig:
        return ModelConfig(
            action_dim=self._action_dim,
            chunk_size=CHUNK_SIZE,
            cameras=list(self.video_keys),
            state_dim=self._state_dim,
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def is_stub(self) -> bool:
        return self.stub

    def close(self) -> None:
        """Release the ZMQ socket and context. Idempotent.

        Without this, a leaked context blocks forever in zmq's
        Context.__del__ -> term() while its socket is still open.
        """
        self._loaded = False
        if self._socket is not None:
            try:
                self._socket.close(linger=0)
            except Exception:
                pass
            self._socket = None
        if self._zmq_ctx is not None:
            try:
                self._zmq_ctx.term()
            except Exception:
                pass
            self._zmq_ctx = None

    # ── ZMQ transport ────────────────────────────────────────────

    def _connect(self) -> None:
        import zmq

        if self._zmq_ctx is None:
            self._zmq_ctx = zmq.Context()
        self._socket = self._zmq_ctx.socket(zmq.REQ)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self._socket.connect(f"tcp://{self.host}:{self.port}")

    def _reconnect(self) -> None:
        # A REQ socket is dead after a failed send/recv (strict lockstep) —
        # it must be rebuilt, not reused.
        if self._socket is not None:
            try:
                self._socket.close(linger=0)
            except Exception:
                pass
            self._socket = None
        try:
            self._connect()
        except Exception as e:
            logger.error(f"GR00T reconnect failed: {e}")

    def _request(self, endpoint: str, data: dict | None = None):
        """One PolicyServer round-trip; rebuilds the socket on failure."""
        import msgpack
        import msgpack_numpy as m

        payload: dict = {"endpoint": endpoint}
        if data is not None:
            payload["data"] = data
        if self.api_token:
            payload["api_token"] = self.api_token

        try:
            self._socket.send(
                msgpack.packb(payload, default=m.encode, use_bin_type=True)
            )
            raw = self._socket.recv()
        except Exception as e:
            self._reconnect()
            raise RuntimeError(
                f"GR00T PolicyServer at {self.host}:{self.port} not responding: {e}"
            )

        result = msgpack.unpackb(raw, object_hook=m.decode, raw=False)
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(f"GR00T server error: {result['error']}")
        return result

    # ── Inference ────────────────────────────────────────────────

    def _zmq_predict(
        self,
        images: dict[str, str],
        state: list[float],
        task: str,
    ) -> list[list[float]]:
        obs = self._build_observation(images, state, task)
        result = self._request("get_action", {"observation": obs})

        # get_action returns (action, info); msgpack delivers it as a list
        action = result[0] if isinstance(result, (list, tuple)) else result
        if not isinstance(action, dict):
            raise RuntimeError(f"Unexpected GR00T response type: {type(action)}")
        return self._parse_action(action)

    # ── Observation building ─────────────────────────────────────

    def _build_observation(
        self,
        images: dict[str, str],
        state: list[float],
        task: str,
    ) -> dict:
        """Convert vla-server format to a batched N1.7 observation dict."""
        obs: dict = {
            "video": {},
            "state": {},
            "language": {self.language_key: [[task]]},
        }

        for key in self.video_keys:
            if key not in images:
                continue
            img_bytes = base64.b64decode(images[key])
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            if self.image_size is not None:
                img = img.resize((self.image_size, self.image_size))
            # (H, W, 3) -> (1, 1, H, W, 3): batch + time dims
            obs["video"][key] = np.array(img, dtype=np.uint8)[None, None]

        # Never fabricate a robot pose: an empty state must fail loudly
        # instead of silently becoming an all-zero joint configuration.
        if not state:
            raise ValueError(
                "state must be a non-empty list of joint positions "
                f"(expected {self._state_dim} values)"
            )
        if len(state) < self._state_dim and not self._warned_partial_state:
            self._warned_partial_state = True
            logger.warning(
                f"state has {len(state)} values, expected {self._state_dim}; "
                f"zero-padding missing joints (logged once)"
            )
        padded = list(state) + [0.0] * max(0, self._state_dim - len(state))
        offset = 0
        for name, dim in self.state_keys.items():
            values = padded[offset : offset + dim]
            obs["state"][name] = np.array(values, dtype=np.float32)[None, None]
            offset += dim

        return obs

    # ── Action parsing ───────────────────────────────────────────

    def _parse_action(self, action: dict) -> list[list[float]]:
        """Concatenate (B, T, D) action arrays into (horizon, action_dim) rows.

        Each key's width is validated against action_keys — a checkpoint
        returning unexpected dims must error, not silently mis-align
        joint commands.
        """
        parts: list[np.ndarray] = []
        for key, dim in self.action_keys.items():
            if key not in action:
                raise RuntimeError(
                    f"GR00T response missing action key '{key}' "
                    f"(got {sorted(action.keys())})"
                )
            arr = np.asarray(action[key], dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[0]  # first batch
            elif arr.ndim == 1:
                # Disambiguate a squeezed 1-D array by the known width:
                # (T,) for a 1-dim action, (D,) for a single step otherwise
                arr = arr.reshape(-1, 1) if dim == 1 else arr.reshape(1, -1)
            if arr.ndim != 2 or arr.shape[1] != dim:
                raise RuntimeError(
                    f"GR00T action '{key}' has shape "
                    f"{np.asarray(action[key]).shape}, expected (*, {dim})"
                )
            parts.append(arr)

        horizon = min(p.shape[0] for p in parts)
        combined = np.concatenate([p[:horizon] for p in parts], axis=1)
        return combined.tolist()

    # ── Stub ─────────────────────────────────────────────────────

    def _stub_predict(self, state: list[float]) -> list[list[float]]:
        """Stub: returns gentle sine-wave actions for testing."""
        actions: list[list[float]] = []
        for i in range(CHUNK_SIZE):
            t = (self._step + i) / 30.0
            action = [
                (state[j] if j < len(state) else 0.0)
                + 2.0 * math.sin(2.0 * math.pi * 0.1 * t + j * 0.5)
                for j in range(self._action_dim)
            ]
            actions.append(action)
        self._step += CHUNK_SIZE
        return actions
