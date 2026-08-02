"""
@file lingbot.py
@description LingBot-VLA 2.0 (Robbyant / Ant Group) backend for vla-server.

Connects to the LingBot deploy server
(``lingbot-vla-v2/deploy/lingbot_vla_v2_policy.py``), which speaks msgpack
over a **WebSocket** — not ZMQ like GR00T.

Wire protocol (``deploy/websocket_policy_server.py``):
- On connect the server pushes one msgpack frame of metadata; it MUST be
  consumed before the first request or every later recv() is off by one.
- Request/response are single msgpack frames in strict lockstep.
- NumPy arrays travel in the openpi encoding (``deploy/msgpack_numpy.py``):
  ``{b"__ndarray__": True, b"data": bytes, b"dtype": str, b"shape": tuple}``.
  This is NOT the ``msgpack-numpy`` PyPI format used by the GR00T backend —
  the two are wire-incompatible, hence the small vendored codec below.

Two request shapes exist (``lingbot_vla_v2_policy.py:461-527``):
- ``{"reset": True, "robo_name": <str>}`` → ``{"action": None}``. This
  rebuilds the FeatureTransform from ``configs/robot_configs/<robo_name>.yaml``
  and re-reads the norm stats, so it must be sent once before the first
  inference and again between episodes.
- an observation dict → ``{"action": (chunk, dim) float32, "server_timing": {...}}``.
  Started with ``--chunk_ret true --use_length -1`` the server re-runs the
  model on every call and returns the full chunk.

Observation keys are NOT fixed by the server: they are the ``origin_keys`` of
the robot config. For ``g1_dex3_apple.yaml`` those are ``observation.state``
(43-dim float32) and ``observation.images.ego_view`` (H, W, 3 uint8), plus
``task``.

**Action order differs from our contract.** ``reverse_features``
(``lingbotvla/data/vla_data/utils.py:359-366``) re-assembles the response
sorted by the *origin* ``end`` index, which yields

    [waist 3 | left_arm 7 | right_arm 7 | left_hand 7 | right_hand 7]

while ``CONTRACT.md`` / every other backend here uses

    [left_arm 7 | right_arm 7 | left_hand 7 | right_hand 7 | waist 3]

The permutation is applied here, driven by ``response_keys`` → ``action_keys``,
so the reordering is declared in the config rather than buried in a slice.
Both orders were confirmed empirically against a real dataset frame: the two
arm blocks track ``observation.state[15:22]`` / ``[22:29]``, and the trailing
block is identically zero — which only ``right_hand`` is in this dataset.

Stub mode returns sine-wave actions without any network or ML deps.
"""

import base64
import io
import logging
import math
import threading
import time

import numpy as np
from PIL import Image

from .base import ModelConfig, PredictResult, VLAModel

logger = logging.getLogger(__name__)

CHUNK_SIZE = 50  # LingBot-VLA 2.0 action horizon (train.chunk_size)

# Response order of the LingBot server, then our contract order. Same names,
# same widths — a pure permutation, validated at load().
DEFAULT_RESPONSE_KEYS = {
    "waist": 3,
    "left_arm": 7,
    "right_arm": 7,
    "left_hand": 7,
    "right_hand": 7,
}
DEFAULT_ACTION_KEYS = {
    "left_arm": 7,
    "right_arm": 7,
    "left_hand": 7,
    "right_hand": 7,
    "waist": 3,
}


# ── msgpack codec (openpi flavour, vendored) ─────────────────────
#
# Kept in-file on purpose: the LingBot repo is a gitignored upstream clone
# whose path differs per host (and is a Linux path when the policy server
# runs in WSL), so importing deploy.msgpack_numpy would tie this backend to
# a checkout that may not exist. ~20 lines, no fallback to pickle.

def _pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"]
        )
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


class LingBotModel(VLAModel):
    """LingBot-VLA 2.0 policy via WebSocket to a LingBot deploy server."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8006,
        stub: bool = False,
        api_key: str | None = None,
        robo_name: str = "g1_dex3_apple",
        image_key: str = "ego_view",
        obs_image_key: str = "observation.images.ego_view",
        obs_state_key: str = "observation.state",
        obs_task_key: str = "task",
        image_size: int | None = None,
        state_dim: int = 43,
        response_keys: dict[str, int] | None = None,
        action_keys: dict[str, int] | None = None,
        chunk_size: int = CHUNK_SIZE,
        timeout_s: float = 120.0,
        connect_retries: int = 3,
    ):
        self.host = host
        self.port = port
        self.stub = stub
        self.api_key = api_key
        # Selects configs/robot_configs/<robo_name>.yaml on the server side.
        self.robo_name = robo_name
        # HTTP-side camera name (what /predict clients send) and the
        # observation key the robot config declares as its origin_key.
        self.image_key = image_key
        self.obs_image_key = obs_image_key
        self.obs_state_key = obs_state_key
        self.obs_task_key = obs_task_key
        # Square resize edge; None sends native resolution (the policy's own
        # processor resizes to data.img_size anyway).
        self.image_size = image_size
        self.state_dim = state_dim
        self.response_keys = dict(response_keys or DEFAULT_RESPONSE_KEYS)
        self.action_keys = dict(action_keys or DEFAULT_ACTION_KEYS)
        self.chunk_size = chunk_size
        self.timeout_s = timeout_s
        self.connect_retries = max(1, connect_retries)
        self._ws = None
        self._packer = None
        self._server_metadata: dict = {}
        self._loaded = False
        self._step = 0
        self._lock = threading.Lock()
        self._warned_partial_state = False

    @property
    def _action_dim(self) -> int:
        return sum(self.action_keys.values())

    @property
    def _response_dim(self) -> int:
        return sum(self.response_keys.values())

    # ── Lifecycle ────────────────────────────────────────────────

    def load(self) -> None:
        if self._loaded:
            logger.warning("LingBotModel already loaded, skipping")
            return

        # A silent mismatch here would mis-align every joint command, so it
        # is checked before a single frame is sent.
        if set(self.response_keys) != set(self.action_keys):
            raise ValueError(
                "lingbot response_keys and action_keys must contain the same "
                f"names (response={sorted(self.response_keys)}, "
                f"action={sorted(self.action_keys)})"
            )
        for name, dim in self.action_keys.items():
            if self.response_keys[name] != dim:
                raise ValueError(
                    f"lingbot key '{name}' has width {self.response_keys[name]} in "
                    f"response_keys but {dim} in action_keys"
                )

        if self.stub:
            logger.info("LingBotModel loaded (stub mode — sine-wave actions)")
            self._loaded = True
            return

        try:
            import msgpack  # noqa: F401
            import websockets.sync.client  # noqa: F401
        except ImportError:
            raise ImportError(
                "LingBot backend requires the lingbot extras: "
                "uv pip install -e '.[lingbot]'"
            )

        last_err: Exception | None = None
        for attempt in range(1, self.connect_retries + 1):
            try:
                self._connect()
                break
            except Exception as e:
                last_err = e
                if attempt < self.connect_retries:
                    logger.warning(
                        f"LingBot connect {attempt}/{self.connect_retries} failed: "
                        f"{e}; retrying in 2s"
                    )
                    time.sleep(2.0)
        if self._ws is None:
            raise RuntimeError(
                f"LingBot server at ws://{self.host}:{self.port} unreachable "
                f"after {self.connect_retries} attempts: {last_err}"
            )
        logger.info(f"LingBot server metadata: {self._server_metadata}")

        # The server has no FeatureTransform until the first reset, so this
        # is mandatory rather than merely tidy — and it fails loudly here if
        # robo_name has no matching robot config on the server.
        self._reset_remote()

        self._loaded = True
        logger.info(
            f"LingBotModel loaded: ws://{self.host}:{self.port} "
            f"(robo_name={self.robo_name}, chunk={self.chunk_size})"
        )

    def reset(self) -> None:
        self._step = 0
        if self._ws is not None and not self.stub:
            try:
                self._reset_remote()
            except Exception as e:
                logger.warning(f"LingBot reset failed (best-effort): {e}")

    def info(self) -> ModelConfig:
        return ModelConfig(
            action_dim=self._action_dim,
            chunk_size=self.chunk_size,
            cameras=[self.image_key],
            state_dim=self.state_dim,
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def is_stub(self) -> bool:
        return self.stub

    def close(self) -> None:
        self._loaded = False
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    # ── Inference ────────────────────────────────────────────────

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
            actions = self._ws_predict(images, state, task)
        return PredictResult(
            actions=actions,
            inference_time_ms=(time.perf_counter() - t_start) * 1000,
        )

    def _ws_predict(
        self,
        images: dict[str, str],
        state: list[float],
        task: str,
    ) -> list[list[float]]:
        obs = self._build_observation(images, state, task)
        result = self._request(obs)

        if not isinstance(result, dict) or "action" not in result:
            raise RuntimeError(
                f"Unexpected LingBot response: {type(result)} "
                f"{sorted(result) if isinstance(result, dict) else ''}"
            )
        action = result["action"]
        if action is None:
            raise RuntimeError(
                "LingBot returned action=None — the request was interpreted as "
                "a reset, not an observation"
            )
        self._step += 1
        return self._reorder(np.asarray(action, dtype=np.float32))

    def _build_observation(
        self,
        images: dict[str, str],
        state: list[float],
        task: str,
    ) -> dict:
        """Convert the vla-server /predict payload into a LingBot observation."""
        if self.image_key not in images:
            raise ValueError(
                f"missing camera '{self.image_key}' in request "
                f"(got {sorted(images)})"
            )
        img = Image.open(io.BytesIO(base64.b64decode(images[self.image_key])))
        img = img.convert("RGB")
        if self.image_size is not None:
            img = img.resize((self.image_size, self.image_size))

        # Never fabricate a robot pose: an empty state must fail loudly
        # instead of silently becoming an all-zero joint configuration.
        if not state:
            raise ValueError(
                "state must be a non-empty list of joint positions "
                f"(expected {self.state_dim} values)"
            )
        if len(state) != self.state_dim and not self._warned_partial_state:
            self._warned_partial_state = True
            logger.warning(
                f"state has {len(state)} values, expected {self.state_dim}; "
                f"padding/truncating (logged once)"
            )
        padded = (list(state) + [0.0] * self.state_dim)[: self.state_dim]

        return {
            self.obs_state_key: np.asarray(padded, dtype=np.float32),
            self.obs_image_key: np.asarray(img, dtype=np.uint8),
            self.obs_task_key: task,
        }

    def _reorder(self, action: np.ndarray) -> list[list[float]]:
        """Permute the server's group order into our contract order.

        Widths are checked against response_keys first: a checkpoint whose
        robot config declares different groups must error rather than
        silently shift every joint command by a few indices.
        """
        if action.ndim == 1:
            action = action[None]
        if action.ndim != 2 or action.shape[1] != self._response_dim:
            raise RuntimeError(
                f"LingBot action has shape {action.shape}, expected "
                f"(*, {self._response_dim}) for groups "
                f"{list(self.response_keys)}"
            )

        offset = 0
        blocks: dict[str, np.ndarray] = {}
        for name, dim in self.response_keys.items():
            blocks[name] = action[:, offset : offset + dim]
            offset += dim
        return np.concatenate(
            [blocks[name] for name in self.action_keys], axis=1
        ).tolist()

    # ── WebSocket transport ──────────────────────────────────────

    def _connect(self) -> None:
        import msgpack
        import websockets.sync.client

        self._packer = msgpack.Packer(default=_pack_array)
        headers = {"Authorization": f"Api-Key {self.api_key}"} if self.api_key else None
        # compression/max_size mirror deploy/websocket_client_policy.py.
        # proxy=None: websockets>=14 honours HTTP(S)_PROXY by default, which
        # would route a localhost connection through a corporate proxy.
        # ping_interval=None disables keepalive — a long inference must not
        # be torn down as an unanswered ping.
        self._ws = websockets.sync.client.connect(
            f"ws://{self.host}:{self.port}",
            compression=None,
            max_size=None,
            additional_headers=headers,
            proxy=None,
            open_timeout=self.timeout_s,
            ping_interval=None,
            ping_timeout=None,
        )
        # The server pushes metadata immediately after the handshake; leaving
        # it in the queue would offset every subsequent response by one frame.
        self._server_metadata = self._recv()

    def _recv(self):
        import msgpack

        raw = self._ws.recv(timeout=self.timeout_s)
        if isinstance(raw, str):
            raise RuntimeError(f"LingBot server error: {raw}")
        return msgpack.unpackb(raw, object_hook=_unpack_array, raw=False)

    def _request(self, payload: dict):
        """One request/response round-trip. Drops the connection on failure.

        The WebSocket is in strict lockstep like a ZMQ REQ socket: after a
        failed send or recv the frame boundary is unknown, so the connection
        is discarded rather than reused. Reconnecting also loses the server's
        FeatureTransform, hence the reset that follows it.
        """
        with self._lock:
            if self._ws is None:
                raise RuntimeError("LingBot connection is closed")
            try:
                self._ws.send(self._packer.pack(payload))
                return self._recv()
            except Exception as e:
                self._drop()
                raise RuntimeError(
                    f"LingBot server at ws://{self.host}:{self.port} "
                    f"not responding: {e}"
                )

    def _drop(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    def _reset_remote(self) -> None:
        result = self._request({"reset": True, "robo_name": self.robo_name})
        if not isinstance(result, dict) or result.get("action") is not None:
            raise RuntimeError(f"Unexpected LingBot reset response: {result}")

    # ── Stub ─────────────────────────────────────────────────────

    def _stub_predict(self, state: list[float]) -> list[list[float]]:
        """Stub: returns gentle sine-wave actions for testing."""
        actions: list[list[float]] = []
        for i in range(self.chunk_size):
            t = (self._step + i) / 30.0
            actions.append(
                [
                    (state[j] if j < len(state) else 0.0)
                    + 2.0 * math.sin(2.0 * math.pi * 0.1 * t + j * 0.5)
                    for j in range(self._action_dim)
                ]
            )
        self._step += self.chunk_size
        return actions
