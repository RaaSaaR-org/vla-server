"""
Tests for the VLA inference server.

Uses the pi05 stub model (no ML dependencies required).
"""

import base64
import io
import time

import pytest
from fastapi.testclient import TestClient

# Patch the server config before importing
from server import ServerConfig, app, create_model


@pytest.fixture
def client():
    """Create a test client with stub model."""
    cfg = ServerConfig(model="pi05", stub=True, port=9999)
    app.state.config = cfg
    with TestClient(app) as c:
        yield c


@pytest.fixture
def dummy_image_b64() -> str:
    """Create a minimal valid JPEG as base64."""
    from PIL import Image

    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True

    def test_health_has_device(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "device" in data


class TestConfig:
    def test_config_returns_dims(self, client):
        resp = client.get("/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["action_dim"] == 6
        assert data["chunk_size"] == 50
        assert "front" in data["cameras"]
        assert data["state_dim"] == 6


class TestPredict:
    def test_predict_returns_actions(self, client, dummy_image_b64):
        resp = client.post("/predict", json={
            "images": {"front": dummy_image_b64},
            "state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "task": "pick up the object",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "actions" in data
        assert len(data["actions"]) == 50  # pi05 chunk_size
        assert len(data["actions"][0]) == 6  # action_dim
        assert "timestamp" in data
        assert "inference_time_ms" in data
        assert data["inference_time_ms"] >= 0

    def test_predict_missing_camera(self, client):
        resp = client.post("/predict", json={
            "images": {},
            "state": [0.0] * 6,
            "task": "test",
        })
        assert resp.status_code == 422

    def test_predict_multiple_calls(self, client, dummy_image_b64):
        """Verify actions change between calls (not constant)."""
        payload = {
            "images": {"front": dummy_image_b64},
            "state": [0.0] * 6,
            "task": "pick up",
        }
        resp1 = client.post("/predict", json=payload)
        resp2 = client.post("/predict", json=payload)
        a1 = resp1.json()["actions"][0]
        a2 = resp2.json()["actions"][0]
        # Sine-wave stub advances step counter, so actions differ
        assert a1 != a2


class TestReset:
    def test_reset_ok(self, client):
        resp = client.post("/reset")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_reset_then_predict_restarts(self, client, dummy_image_b64):
        """After reset, stub should produce same initial actions."""
        payload = {
            "images": {"front": dummy_image_b64},
            "state": [0.0] * 6,
            "task": "test",
        }
        resp1 = client.post("/predict", json=payload)
        client.post("/reset")
        resp2 = client.post("/predict", json=payload)
        assert resp1.json()["actions"] == resp2.json()["actions"]


class TestAuth:
    TOKEN = "test-service-token"

    @pytest.fixture
    def auth_client(self):
        cfg = ServerConfig(model="pi05", stub=True, port=9999, auth_token=self.TOKEN)
        app.state.config = cfg
        with TestClient(app) as c:
            yield c

    def _headers(self, token: str | None = None) -> dict:
        return {"Authorization": f"Bearer {token or self.TOKEN}"}

    def test_health_open_without_token(self, auth_client):
        resp = auth_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["auth_enabled"] is True

    def test_predict_requires_token(self, auth_client, dummy_image_b64):
        resp = auth_client.post("/predict", json={
            "images": {"front": dummy_image_b64},
            "state": [0.0] * 6,
            "task": "test",
        })
        assert resp.status_code == 401
        assert resp.headers["WWW-Authenticate"] == "Bearer"

    def test_predict_with_valid_token(self, auth_client, dummy_image_b64):
        resp = auth_client.post("/predict", json={
            "images": {"front": dummy_image_b64},
            "state": [0.0] * 6,
            "task": "test",
        }, headers=self._headers())
        assert resp.status_code == 200

    def test_wrong_token_rejected(self, auth_client, dummy_image_b64):
        resp = auth_client.post("/predict", json={
            "images": {"front": dummy_image_b64},
            "state": [0.0] * 6,
            "task": "test",
        }, headers=self._headers("wrong-token"))
        assert resp.status_code == 403

    def test_config_requires_token(self, auth_client):
        assert auth_client.get("/config").status_code == 401
        assert auth_client.get("/config", headers=self._headers()).status_code == 200

    def test_reset_requires_token(self, auth_client):
        assert auth_client.post("/reset").status_code == 401
        assert auth_client.post("/reset", headers=self._headers()).status_code == 200

    def test_load_adapter_requires_token(self, auth_client):
        resp = auth_client.post("/load-adapter", json={"adapter_path": "/x"})
        assert resp.status_code == 401

    def test_no_token_configured_means_open(self, client, dummy_image_b64):
        """Without auth_token, endpoints stay open (dev mode)."""
        resp = client.post("/predict", json={
            "images": {"front": dummy_image_b64},
            "state": [0.0] * 6,
            "task": "test",
        })
        assert resp.status_code == 200
        assert client.get("/health").json()["auth_enabled"] is False

    def test_health_reports_stub(self, client):
        assert client.get("/health").json()["stub"] is True


class TestModelFactory:
    def test_create_stub_model(self):
        cfg = ServerConfig(stub=True)
        model = create_model(cfg)
        from models.pi05 import Pi05Model
        assert isinstance(model, Pi05Model)

    def test_create_pi05_model(self):
        cfg = ServerConfig(model="pi05")
        model = create_model(cfg)
        from models.pi05 import Pi05Model
        assert isinstance(model, Pi05Model)

    def test_create_unknown_model_raises(self):
        cfg = ServerConfig(model="unknown", stub=False)
        with pytest.raises(ValueError, match="Unknown model"):
            create_model(cfg)


class TestPredictMultiCamera:
    def test_predict_with_images_dict(self, client, dummy_image_b64):
        """POST with images dict containing front + wrist cameras."""
        resp = client.post("/predict", json={
            "images": {"front": dummy_image_b64, "wrist": dummy_image_b64},
            "state": [0.0] * 6,
            "task": "pick up the cube",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "actions" in data
        assert len(data["actions"]) == 50

    def test_predict_backward_compat_image_b64(self, client, dummy_image_b64):
        """POST with legacy image_b64 field (no images dict)."""
        resp = client.post("/predict", json={
            "image_b64": dummy_image_b64,
            "state": [0.0] * 6,
            "task": "pick up the cube",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "actions" in data
        assert len(data["actions"]) == 50
        assert len(data["actions"][0]) == 6

    def test_predict_no_images_fails(self, client):
        """POST with neither images nor image_b64 → 422."""
        resp = client.post("/predict", json={
            "state": [0.0] * 6,
            "task": "test",
        })
        assert resp.status_code == 422

    def test_predict_images_takes_precedence(self, client, dummy_image_b64):
        """When both images and image_b64 provided, images wins."""
        resp = client.post("/predict", json={
            "image_b64": "should_be_ignored",
            "images": {"front": dummy_image_b64},
            "state": [0.0] * 6,
            "task": "test",
        })
        assert resp.status_code == 200
