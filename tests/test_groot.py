"""
Tests for GR00T N1.7 model backend.

Stub and parsing tests need no ZMQ deps; wire-protocol tests run a minimal
in-process Isaac-GR00T PolicyServer emulator over real ZMQ (loopback) and
are skipped when the groot extras are not installed.
"""

import base64
import io
import threading
import time

import numpy as np
import pytest
from PIL import Image

from models.groot import ACTION_DIM, CHUNK_SIZE, GR00TModel

try:
    import msgpack
    import msgpack_numpy as msgpack_np
    import zmq
except ImportError:
    zmq = None

requires_zmq = pytest.mark.skipif(
    zmq is None, reason="pyzmq/msgpack not installed (groot extras)"
)


@pytest.fixture
def stub_model() -> GR00TModel:
    """GR00T model in stub mode, loaded and ready."""
    model = GR00TModel(stub=True)
    model.load()
    return model


def make_image_b64(width: int, height: int, color: tuple[int, int, int]) -> str:
    """Create a solid-color JPEG of the given size as base64."""
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def dummy_image_b64() -> str:
    """Create a minimal valid JPEG as base64."""
    return make_image_b64(64, 64, (128, 128, 128))


# ── Fake Isaac-GR00T PolicyServer ────────────────────────────────


class FakeGrootServer(threading.Thread):
    """Emulates the Isaac-GR00T N1.7 PolicyServer wire protocol.

    Envelope: {"endpoint", "data", "api_token"?} via msgpack_numpy.
    get_action returns (action_dict, info_dict); errors as {"error": str}.
    """

    def __init__(
        self,
        api_token: str | None = None,
        horizon: int = CHUNK_SIZE,
        action_factory=None,
    ):
        super().__init__(daemon=True)
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.REP)
        self.port = self.socket.bind_to_random_port("tcp://127.0.0.1")
        self.api_token = api_token
        self.horizon = horizon
        # Optional () -> action_dict override for get_action responses
        # (e.g. embodiments other than the SO-101 default below).
        self.action_factory = action_factory
        self.requests: list[dict] = []
        self.delay_next_s: float = 0.0
        self._stop_event = threading.Event()

    def run(self):
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        while not self._stop_event.is_set():
            if not poller.poll(50):
                continue
            raw = self.socket.recv()
            req = msgpack.unpackb(raw, object_hook=msgpack_np.decode, raw=False)
            self.requests.append(req)
            if self.delay_next_s:
                time.sleep(self.delay_next_s)  # outlive the client's RCVTIMEO
                self.delay_next_s = 0.0
            self.socket.send(
                msgpack.packb(
                    self._respond(req), default=msgpack_np.encode, use_bin_type=True
                )
            )
        self.socket.close(linger=0)
        self.ctx.term()

    def _respond(self, req: dict):
        if self.api_token and req.get("api_token") != self.api_token:
            return {"error": "unauthorized"}
        endpoint = req.get("endpoint")
        if endpoint == "ping":
            return {"status": "ok", "message": "Server is running"}
        if endpoint == "reset":
            return {"status": "reset"}
        if endpoint == "get_action":
            if self.action_factory is not None:
                return [self.action_factory(), {}]
            action = {
                "single_arm": np.arange(self.horizon * 5, dtype=np.float32).reshape(
                    1, self.horizon, 5
                ),
                "gripper": np.full((1, self.horizon, 1), 0.5, dtype=np.float32),
            }
            return [action, {}]  # (action, info) tuple arrives as list
        return {"error": f"unknown endpoint: {endpoint}"}

    def stop(self):
        self._stop_event.set()
        self.join(timeout=2)


@pytest.fixture
def fake_server():
    server = FakeGrootServer()
    server.start()
    yield server
    server.stop()


# ── Stub-mode tests ──────────────────────────────────────────────


class TestGR00TStubPredict:
    def test_stub_returns_correct_shape(self, stub_model):
        result = stub_model.predict(
            images={},
            state=[0.0] * 6,
            task="pick up the object",
        )
        assert len(result.actions) == CHUNK_SIZE
        assert all(len(a) == ACTION_DIM for a in result.actions)

    def test_stub_inference_time_positive(self, stub_model):
        result = stub_model.predict(images={}, state=[0.0] * 6, task="test")
        assert result.inference_time_ms >= 0

    def test_stub_actions_vary_between_calls(self, stub_model):
        r1 = stub_model.predict(images={}, state=[0.0] * 6, task="test")
        r2 = stub_model.predict(images={}, state=[0.0] * 6, task="test")
        assert r1.actions[0] != r2.actions[0]

    def test_stub_reports_is_stub(self, stub_model):
        assert stub_model.is_stub is True


class TestGR00TInfo:
    def test_info_defaults(self, stub_model):
        info = stub_model.info()
        assert info.action_dim == ACTION_DIM
        assert info.chunk_size == CHUNK_SIZE
        assert info.state_dim == 6
        assert "front" in info.cameras

    def test_info_custom_modality(self):
        model = GR00TModel(
            stub=True,
            video_key="ego_view",
            state_keys={"left_arm": 7, "right_arm": 7},
            action_keys={"left_arm": 7, "right_arm": 7},
        )
        info = model.info()
        assert info.cameras == ["ego_view"]
        assert info.state_dim == 14
        assert info.action_dim == 14

    def test_info_cameras_multi(self):
        model = GR00TModel(
            stub=True, video_keys=["cam_left_high", "cam_right_high"]
        )
        info = model.info()
        assert info.cameras == ["cam_left_high", "cam_right_high"]

    def test_video_keys_wins_over_video_key(self):
        model = GR00TModel(
            stub=True, video_key="front", video_keys=["cam_right_high"]
        )
        assert model.video_keys == ["cam_right_high"]
        assert model.video_key == "cam_right_high"  # back-compat alias
        assert model.info().cameras == ["cam_right_high"]

    def test_single_video_key_back_compat(self):
        """Legacy single video_key constructor arg is unchanged."""
        model = GR00TModel(stub=True, video_key="ego_view")
        assert model.video_key == "ego_view"
        assert model.video_keys == ["ego_view"]
        assert model.info().cameras == ["ego_view"]

    def test_info_action_dim_independent_of_state(self):
        """Extra state-only keys must not inflate action_dim."""
        model = GR00TModel(
            stub=True,
            state_keys={"single_arm": 5, "gripper": 1, "base_pose": 3},
            action_keys={"single_arm": 5, "gripper": 1},
        )
        info = model.info()
        assert info.state_dim == 9
        assert info.action_dim == 6


class TestGR00TLoadState:
    def test_not_loaded_before_load(self):
        model = GR00TModel(stub=True)
        assert model.is_loaded is False

    def test_loaded_after_load(self):
        model = GR00TModel(stub=True)
        model.load()
        assert model.is_loaded is True

    def test_predict_before_load_raises(self):
        model = GR00TModel(stub=True)
        with pytest.raises(RuntimeError, match="not loaded"):
            model.predict(images={}, state=[0.0] * 6, task="test")

    def test_double_load_warns(self, stub_model, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            stub_model.load()
        assert "already loaded" in caplog.text

    def test_reset_restores_step(self, stub_model):
        stub_model.predict(images={}, state=[0.0] * 6, task="test")
        assert stub_model._step > 0
        stub_model.reset()
        assert stub_model._step == 0


# ── Observation building (N1.7 nested format) ────────────────────


class TestGR00TBuildObservation:
    def test_observation_video_batched(self, stub_model, dummy_image_b64):
        obs = stub_model._build_observation(
            images={"front": dummy_image_b64},
            state=[1.0, 2.0, 3.0, 4.0, 5.0, 0.5],
            task="pick up the cube",
        )
        assert obs["video"]["front"].shape == (1, 1, 224, 224, 3)
        assert obs["video"]["front"].dtype == np.uint8

    def test_observation_state_split_batched(self, stub_model, dummy_image_b64):
        obs = stub_model._build_observation(
            images={"front": dummy_image_b64},
            state=[1.0, 2.0, 3.0, 4.0, 5.0, 0.5],
            task="test",
        )
        assert obs["state"]["single_arm"].shape == (1, 1, 5)
        assert obs["state"]["gripper"].shape == (1, 1, 1)
        np.testing.assert_array_equal(
            obs["state"]["single_arm"][0, 0],
            np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            obs["state"]["gripper"][0, 0], np.array([0.5], dtype=np.float32)
        )

    def test_observation_language_task(self, stub_model, dummy_image_b64):
        obs = stub_model._build_observation(
            images={"front": dummy_image_b64},
            state=[0.0] * 6,
            task="move left",
        )
        assert obs["language"]["task"] == [["move left"]]

    def test_observation_partial_state_zero_padded(self, stub_model):
        obs = stub_model._build_observation(images={}, state=[1.0, 2.0], task="test")
        np.testing.assert_array_equal(
            obs["state"]["single_arm"][0, 0],
            np.array([1.0, 2.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            obs["state"]["gripper"][0, 0], np.array([0.0], dtype=np.float32)
        )

    def test_observation_empty_state_raises(self, stub_model):
        """An empty state must never become a fabricated all-zero pose."""
        with pytest.raises(ValueError, match="non-empty"):
            stub_model._build_observation(images={}, state=[], task="test")

    def test_observation_no_image(self, stub_model):
        obs = stub_model._build_observation(images={}, state=[0.0] * 6, task="test")
        assert obs["video"] == {}
        assert "single_arm" in obs["state"]

    def test_observation_multi_camera_independent_decode(self):
        """Every configured camera is forwarded, each decoded on its own."""
        model = GR00TModel(
            stub=True, video_keys=["cam_left_high", "cam_right_high"]
        )
        model.load()
        images = {
            "cam_left_high": make_image_b64(64, 64, (250, 0, 0)),
            "cam_right_high": make_image_b64(32, 32, (0, 0, 250)),
        }
        obs = model._build_observation(images=images, state=[0.0] * 6, task="t")
        assert set(obs["video"].keys()) == {"cam_left_high", "cam_right_high"}
        left = obs["video"]["cam_left_high"]
        right = obs["video"]["cam_right_high"]
        assert left.shape == (1, 1, 224, 224, 3)
        assert right.shape == (1, 1, 224, 224, 3)
        # Independent decode: red frame vs blue frame, not the same buffer
        assert left[0, 0, 0, 0, 0] > left[0, 0, 0, 0, 2]   # red > blue
        assert right[0, 0, 0, 0, 2] > right[0, 0, 0, 0, 0]  # blue > red

    def test_observation_multi_camera_missing_one_skipped(self):
        model = GR00TModel(
            stub=True, video_keys=["cam_left_high", "cam_right_high"]
        )
        model.load()
        images = {"cam_right_high": make_image_b64(64, 64, (0, 250, 0))}
        obs = model._build_observation(images=images, state=[0.0] * 6, task="t")
        assert list(obs["video"].keys()) == ["cam_right_high"]

    def test_observation_image_size_none_passthrough(self):
        """image_size=None must keep the native resolution (no resize)."""
        model = GR00TModel(
            stub=True, video_keys=["cam_right_high"], image_size=None
        )
        model.load()
        # PIL size is (W, H) = (640, 480) -> array (H, W, 3) = (480, 640, 3)
        images = {"cam_right_high": make_image_b64(640, 480, (10, 20, 30))}
        obs = model._build_observation(images=images, state=[0.0] * 6, task="t")
        assert obs["video"]["cam_right_high"].shape == (1, 1, 480, 640, 3)
        assert obs["video"]["cam_right_high"].dtype == np.uint8

    def test_observation_custom_image_size(self):
        model = GR00TModel(stub=True, image_size=96)
        model.load()
        obs = model._build_observation(
            images={"front": make_image_b64(640, 480, (1, 2, 3))},
            state=[0.0] * 6,
            task="t",
        )
        assert obs["video"]["front"].shape == (1, 1, 96, 96, 3)

    def test_observation_custom_language_key(self):
        model = GR00TModel(
            stub=True, language_key="annotation.human.task_description"
        )
        model.load()
        obs = model._build_observation(
            images={}, state=[0.0] * 6, task="Put the bottle into the plate."
        )
        assert obs["language"] == {
            "annotation.human.task_description": [["Put the bottle into the plate."]]
        }
        assert "task" not in obs["language"]


# ── Action parsing ───────────────────────────────────────────────


class TestGR00TParseAction:
    def test_parse_batched_3d(self, stub_model):
        action = {
            "single_arm": np.ones((1, 4, 5), dtype=np.float32),
            "gripper": np.full((1, 4, 1), 0.5, dtype=np.float32),
        }
        rows = stub_model._parse_action(action)
        assert len(rows) == 4
        assert rows[0] == [1.0, 1.0, 1.0, 1.0, 1.0, 0.5]

    def test_parse_unbatched_2d(self, stub_model):
        action = {
            "single_arm": np.zeros((16, 5), dtype=np.float32),
            "gripper": np.ones((16, 1), dtype=np.float32),
        }
        rows = stub_model._parse_action(action)
        assert len(rows) == 16
        assert rows[0][-1] == 1.0

    def test_parse_missing_key_raises(self, stub_model):
        with pytest.raises(RuntimeError, match="missing action key"):
            stub_model._parse_action({"single_arm": np.zeros((1, 4, 5))})

    def test_parse_wrong_width_raises(self, stub_model):
        """A checkpoint returning unexpected dims must error, not mis-align."""
        action = {
            "single_arm": np.zeros((1, 4, 6), dtype=np.float32),  # 6 != 5
            "gripper": np.zeros((1, 4, 1), dtype=np.float32),
        }
        with pytest.raises(RuntimeError, match="expected"):
            stub_model._parse_action(action)

    def test_parse_extra_keys_ignored_and_order_kept(self, stub_model):
        """Response keys not in action_keys (navigate_command, effort_*, ...)
        must be ignored — only configured keys, in config order, reach the
        flat action."""
        action = {
            # extra keys first so dict order cannot mask a selection bug
            "navigate_command": np.full((1, 4, 3), 9.0, dtype=np.float32),
            "effort_single_arm": np.full((1, 4, 5), 9.0, dtype=np.float32),
            "gripper": np.full((1, 4, 1), 0.5, dtype=np.float32),
            "single_arm": np.ones((1, 4, 5), dtype=np.float32),
            "base_height_command": np.full((1, 4, 1), 9.0, dtype=np.float32),
        }
        rows = stub_model._parse_action(action)
        assert len(rows) == 4
        # config order (single_arm, gripper) wins over response dict order
        assert rows[0] == [1.0, 1.0, 1.0, 1.0, 1.0, 0.5]
        assert all(9.0 not in row for row in rows)

    def test_parse_squeezed_single_step(self, stub_model):
        """A squeezed (D,) single-step arm action is one row, not D rows."""
        action = {
            "single_arm": np.array([1, 2, 3, 4, 5], dtype=np.float32),  # (5,)
            "gripper": np.array([0.5], dtype=np.float32),               # (1,)
        }
        rows = stub_model._parse_action(action)
        assert len(rows) == 1
        assert rows[0] == [1.0, 2.0, 3.0, 4.0, 5.0, 0.5]


# ── Wire-protocol tests against the fake PolicyServer ────────────


@requires_zmq
class TestGR00TProtocol:
    @pytest.fixture(autouse=True)
    def _track_models(self):
        """Close every model after the test — a leaked ZMQ context blocks
        the whole process in Context.__del__ -> term()."""
        self._models: list[GR00TModel] = []
        yield
        for model in self._models:
            model.close()

    def _model(self, server: FakeGrootServer, **kwargs) -> GR00TModel:
        model = GR00TModel(
            host="127.0.0.1", port=server.port, timeout_ms=2000,
            ping_retries=1, **kwargs
        )
        self._models.append(model)
        model.load()
        return model

    def test_load_pings_server(self, fake_server):
        model = self._model(fake_server)
        assert model.is_loaded
        assert fake_server.requests[0]["endpoint"] == "ping"

    def test_load_fails_fast_when_unreachable(self):
        ctx = zmq.Context()
        probe = ctx.socket(zmq.REP)
        free_port = probe.bind_to_random_port("tcp://127.0.0.1")
        probe.close(linger=0)
        ctx.term()

        model = GR00TModel(
            host="127.0.0.1", port=free_port, timeout_ms=300, ping_retries=1
        )
        self._models.append(model)
        with pytest.raises(RuntimeError, match="unreachable"):
            model.load()

    def test_predict_round_trip(self, fake_server, dummy_image_b64):
        model = self._model(fake_server)
        result = model.predict(
            images={"front": dummy_image_b64},
            state=[0.0] * 6,
            task="pick up the cube",
        )
        assert len(result.actions) == CHUNK_SIZE
        assert len(result.actions[0]) == ACTION_DIM
        assert result.actions[0][-1] == 0.5  # gripper from fake server

        req = fake_server.requests[-1]
        assert req["endpoint"] == "get_action"
        obs = req["data"]["observation"]
        assert obs["video"]["front"].shape == (1, 1, 224, 224, 3)
        assert obs["language"]["task"] == [["pick up the cube"]]

    def test_api_token_sent_and_accepted(self, dummy_image_b64):
        server = FakeGrootServer(api_token="sekret")
        server.start()
        try:
            model = self._model(server, api_token="sekret")
            result = model.predict(
                images={"front": dummy_image_b64}, state=[0.0] * 6, task="t"
            )
            assert len(result.actions) == CHUNK_SIZE
            assert all(r.get("api_token") == "sekret" for r in server.requests)
        finally:
            server.stop()

    def test_wrong_api_token_raises(self):
        server = FakeGrootServer(api_token="sekret")
        server.start()
        try:
            model = GR00TModel(
                host="127.0.0.1", port=server.port, timeout_ms=2000,
                api_token="wrong", ping_retries=3,
            )
            self._models.append(model)
            # Auth errors must fail immediately, not burn ping retries
            t0 = time.monotonic()
            with pytest.raises(RuntimeError, match="unauthorized"):
                model.load()  # ping already rejected
            assert time.monotonic() - t0 < 1.5
        finally:
            server.stop()

    def test_language_key_round_trip(self, fake_server, dummy_image_b64):
        """The wire observation must carry the configured language key."""
        model = self._model(
            fake_server,
            language_key="annotation.human.task_description",
        )
        result = model.predict(
            images={"front": dummy_image_b64},
            state=[0.0] * 6,
            task="Put the bottle into the plate.",
        )
        assert len(result.actions) == CHUNK_SIZE

        req = fake_server.requests[-1]
        assert req["endpoint"] == "get_action"
        obs = req["data"]["observation"]
        assert obs["language"]["annotation.human.task_description"] == [
            ["Put the bottle into the plate."]
        ]
        assert "task" not in obs["language"]

    def test_multi_camera_native_resolution_round_trip(self, fake_server):
        """Two cameras at image_size=None arrive unresized on the wire."""
        model = self._model(
            fake_server,
            video_keys=["cam_left_high", "cam_right_high"],
            image_size=None,
        )
        images = {
            "cam_left_high": make_image_b64(640, 480, (250, 0, 0)),
            "cam_right_high": make_image_b64(640, 480, (0, 0, 250)),
        }
        model.predict(images=images, state=[0.0] * 6, task="t")

        obs = fake_server.requests[-1]["data"]["observation"]
        assert obs["video"]["cam_left_high"].shape == (1, 1, 480, 640, 3)
        assert obs["video"]["cam_right_high"].shape == (1, 1, 480, 640, 3)

    def test_reset_sends_reset_endpoint(self, fake_server):
        model = self._model(fake_server)
        model.reset()
        assert fake_server.requests[-1]["endpoint"] == "reset"

    def test_recovers_after_timeout(self, fake_server, dummy_image_b64):
        """A REQ socket is dead after a timeout — verify we rebuild it."""
        model = GR00TModel(
            host="127.0.0.1", port=fake_server.port, timeout_ms=300, ping_retries=1
        )
        self._models.append(model)
        model.load()

        fake_server.delay_next_s = 1.0  # outlives the client's 300ms RCVTIMEO
        with pytest.raises(RuntimeError, match="not responding"):
            model.predict(
                images={"front": dummy_image_b64}, state=[0.0] * 6, task="t"
            )

        time.sleep(1.2)  # let the fake server flush its delayed (dropped) reply
        result = model.predict(
            images={"front": dummy_image_b64}, state=[0.0] * 6, task="t"
        )
        assert len(result.actions) == CHUNK_SIZE


# ── Apple-to-plate contract (g1_apple_pnp) ───────────────────────

# Mirrors configs/g1_apple_pnp.yaml + C:\Unitree\_data\apple_pnp\CONTRACT.md
APPLE_STATE_KEYS = {
    "left_leg": 6,
    "right_leg": 6,
    "waist": 3,
    "left_arm": 7,
    "right_arm": 7,
    "left_hand": 7,
    "right_hand": 7,
}  # 43-dim
APPLE_ACTION_KEYS = {
    "left_arm": 7,
    "right_arm": 7,
    "left_hand": 7,
    "right_hand": 7,
    "waist": 3,
}  # 31-dim
APPLE_TASK = "move the apple to the plate"


def make_apple_action(horizon: int = CHUNK_SIZE) -> dict:
    """PolicyServer response for the apple checkpoint: the 5 configured
    action keys (distinct fill values so concatenation order is provable)
    PLUS the extra keys the real checkpoint also returns."""
    action = {
        # extras interleaved first — dict order must not matter
        "navigate_command": np.full((1, horizon, 3), 9.0, dtype=np.float32),
        "base_height_command": np.full((1, horizon, 1), 9.0, dtype=np.float32),
        "left_arm": np.full((1, horizon, 7), 1.0, dtype=np.float32),
        "effort_left_arm": np.full((1, horizon, 7), 9.0, dtype=np.float32),
        "right_arm": np.full((1, horizon, 7), 2.0, dtype=np.float32),
        "effort_right_arm": np.full((1, horizon, 7), 9.0, dtype=np.float32),
        "left_hand": np.full((1, horizon, 7), 3.0, dtype=np.float32),
        "effort_left_hand": np.full((1, horizon, 7), 9.0, dtype=np.float32),
        "right_hand": np.full((1, horizon, 7), 4.0, dtype=np.float32),
        "effort_right_hand": np.full((1, horizon, 7), 9.0, dtype=np.float32),
        "waist": np.full((1, horizon, 3), 5.0, dtype=np.float32),
        "effort_waist": np.full((1, horizon, 3), 9.0, dtype=np.float32),
    }
    return action


@requires_zmq
class TestGR00TApplePnpContract:
    """End-to-end wire contract for the apple-to-plate embodiment."""

    @pytest.fixture(autouse=True)
    def _track_models(self):
        self._models: list[GR00TModel] = []
        yield
        for model in self._models:
            model.close()

    @pytest.fixture
    def apple_server(self):
        server = FakeGrootServer(action_factory=make_apple_action)
        server.start()
        yield server
        server.stop()

    def _model(self, server: FakeGrootServer) -> GR00TModel:
        model = GR00TModel(
            host="127.0.0.1",
            port=server.port,
            timeout_ms=2000,
            ping_retries=1,
            video_keys=["ego_view"],
            language_key="annotation.human.task_description",
            image_size=None,
            state_keys=APPLE_STATE_KEYS,
            action_keys=APPLE_ACTION_KEYS,
        )
        self._models.append(model)
        model.load()
        return model

    def test_apple_round_trip(self, apple_server):
        model = self._model(apple_server)
        # 43 distinct values so each state.<key> slice is provable
        state = [float(i) for i in range(43)]
        result = model.predict(
            images={"ego_view": make_image_b64(640, 480, (200, 30, 30))},
            state=state,
            task=APPLE_TASK,
        )

        # ── what the PolicyServer received ──
        req = apple_server.requests[-1]
        assert req["endpoint"] == "get_action"
        obs = req["data"]["observation"]
        offset = 0
        for name, dim in APPLE_STATE_KEYS.items():
            assert obs["state"][name].shape == (1, 1, dim), name
            np.testing.assert_array_equal(
                obs["state"][name][0, 0],
                np.arange(offset, offset + dim, dtype=np.float32),
            )
            offset += dim
        assert offset == 43
        # native 640x480 -> array (H, W, 3) = (480, 640, 3), no resize
        assert obs["video"]["ego_view"].shape == (1, 1, 480, 640, 3)
        assert obs["language"] == {
            "annotation.human.task_description": [[APPLE_TASK]]
        }
        assert "task" not in obs["language"]

        # ── what the client returned ──
        assert len(result.actions) == CHUNK_SIZE
        assert all(len(row) == 31 for row in result.actions)
        row = result.actions[0]
        assert row[0:7] == [1.0] * 7      # left_arm
        assert row[7:14] == [2.0] * 7     # right_arm
        assert row[14:21] == [3.0] * 7    # left_hand
        assert row[21:28] == [4.0] * 7    # right_hand
        assert row[28:31] == [5.0] * 3    # waist
        # extra keys (navigate/base_height/effort_*, all 9.0) never leak in
        assert all(9.0 not in r for r in result.actions)

    def test_apple_info_dims(self, apple_server):
        model = self._model(apple_server)
        info = model.info()
        assert info.state_dim == 43
        assert info.action_dim == 31
        assert info.cameras == ["ego_view"]
        assert info.chunk_size == CHUNK_SIZE


# ── Server integration (stub mode) ───────────────────────────────


class TestServerWithGR00T:
    """Integration tests: GR00T model via FastAPI server in stub mode."""

    @pytest.fixture
    def client(self, monkeypatch):
        monkeypatch.setenv("VLA_STUB", "true")
        from server import ServerConfig, app
        from fastapi.testclient import TestClient

        cfg = ServerConfig(model="groot", stub=True, port=9998)
        app.state.config = cfg
        with TestClient(app) as c:
            yield c

    def test_health_groot(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert data["model"] == "groot"
        assert data["stub"] is True

    def test_config_groot(self, client):
        resp = client.get("/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["action_dim"] == ACTION_DIM
        assert data["chunk_size"] == CHUNK_SIZE

    def test_predict_groot_stub(self, client, dummy_image_b64):
        resp = client.post("/predict", json={
            "images": {"front": dummy_image_b64},
            "state": [0.0] * 6,
            "task": "pick up the object",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["actions"]) == CHUNK_SIZE
        assert len(data["actions"][0]) == ACTION_DIM
        assert data["inference_time_ms"] >= 0

    def test_reset_groot(self, client):
        resp = client.post("/reset")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True


class TestModelFactory:
    def test_create_groot_model(self, monkeypatch):
        monkeypatch.setenv("VLA_STUB", "true")
        from server import ServerConfig, create_model

        cfg = ServerConfig(model="groot", stub=True)
        model = create_model(cfg)
        assert isinstance(model, GR00TModel)
        assert model.stub is True

    def test_create_groot_custom_host(self, monkeypatch):
        monkeypatch.setenv("VLA_HOST", "192.168.1.100")
        monkeypatch.setenv("VLA_ZMQ_PORT", "6666")
        monkeypatch.setenv("VLA_STUB", "true")
        from server import ServerConfig, create_model

        cfg = ServerConfig(model="groot")
        model = create_model(cfg)
        assert isinstance(model, GR00TModel)
        assert model.host == "192.168.1.100"
        assert model.port == 6666

    def test_create_groot_from_config(self, monkeypatch):
        monkeypatch.delenv("VLA_HOST", raising=False)
        monkeypatch.delenv("VLA_ZMQ_PORT", raising=False)
        monkeypatch.setenv("VLA_STUB", "true")
        from server import ServerConfig, create_model

        cfg = ServerConfig(
            model="groot",
            groot_host="10.0.0.7",
            groot_port=7777,
            groot_api_token="tok",
            groot_video_key="ego_view",
        )
        model = create_model(cfg)
        assert model.host == "10.0.0.7"
        assert model.port == 7777
        assert model.api_token == "tok"
        assert model.video_key == "ego_view"
        assert model.video_keys == ["ego_view"]
        assert model.language_key == "task"
        assert model.image_size == 224

    def test_create_groot_g1_dex3_fields(self, monkeypatch):
        monkeypatch.delenv("VLA_HOST", raising=False)
        monkeypatch.delenv("VLA_ZMQ_PORT", raising=False)
        monkeypatch.setenv("VLA_STUB", "true")
        from server import ServerConfig, create_model

        cfg = ServerConfig(
            model="groot",
            groot_video_keys=["cam_left_high", "cam_right_high"],
            groot_language_key="annotation.human.task_description",
            groot_image_size=None,
            groot_state_keys={"arms": 14, "hands": 14},
            groot_action_keys={"arms": 14, "hands": 14},
        )
        model = create_model(cfg)
        assert model.video_keys == ["cam_left_high", "cam_right_high"]
        assert model.language_key == "annotation.human.task_description"
        assert model.image_size is None
        info = model.info()
        assert info.cameras == ["cam_left_high", "cam_right_high"]
        assert info.state_dim == 28
        assert info.action_dim == 28


class TestG1Dex3ConfigFiles:
    """The shipped configs/ files must load into the intended contract."""

    @staticmethod
    def _load(name: str):
        from pathlib import Path

        from server import ServerConfig

        path = Path(__file__).resolve().parent.parent / "configs" / name
        assert path.exists(), f"missing config file: {path}"
        return ServerConfig.from_yaml(path)

    def test_1cam_config(self):
        cfg = self._load("g1_dex3_1cam.yaml")
        assert cfg.model == "groot"
        assert cfg.port == 8000
        assert cfg.groot_host == "localhost"
        assert cfg.groot_port == 6555
        assert cfg.groot_video_keys == ["cam_right_high"]
        assert cfg.groot_language_key == "annotation.human.task_description"
        assert cfg.groot_image_size is None  # YAML null = native resolution
        assert cfg.groot_state_keys == {"arms": 14, "hands": 14}
        assert cfg.groot_action_keys == {"arms": 14, "hands": 14}
        assert cfg.default_task == "Put the bottle into the plate."

    def test_2cam_config(self):
        cfg = self._load("g1_dex3_2cam.yaml")
        assert cfg.model == "groot"
        assert cfg.groot_video_keys == ["cam_left_high", "cam_right_high"]
        assert cfg.groot_language_key == "annotation.human.task_description"
        assert cfg.groot_image_size is None
        assert cfg.groot_state_keys == {"arms": 14, "hands": 14}
        assert cfg.groot_action_keys == {"arms": 14, "hands": 14}

    def test_apple_pnp_config(self):
        cfg = self._load("g1_apple_pnp.yaml")
        assert cfg.model == "groot"
        assert cfg.port == 8000
        assert cfg.groot_host == "localhost"
        assert cfg.groot_port == 6555
        assert cfg.groot_video_keys == ["ego_view"]
        assert cfg.groot_language_key == "annotation.human.task_description"
        assert cfg.groot_image_size is None  # native 640x480 passthrough
        assert cfg.groot_state_keys == APPLE_STATE_KEYS
        assert cfg.groot_action_keys == APPLE_ACTION_KEYS
        # dict ORDER is the wire contract — flat state/action slice layout
        assert list(cfg.groot_state_keys) == [
            "left_leg", "right_leg", "waist",
            "left_arm", "right_arm", "left_hand", "right_hand",
        ]
        assert list(cfg.groot_action_keys) == [
            "left_arm", "right_arm", "left_hand", "right_hand", "waist",
        ]
        assert sum(cfg.groot_state_keys.values()) == 43
        assert sum(cfg.groot_action_keys.values()) == 31
        assert cfg.default_task == "move the apple to the plate"
