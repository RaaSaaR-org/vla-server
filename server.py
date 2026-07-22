"""
@file server.py
@description Consolidated VLA inference server — FastAPI HTTP.

Replaces smolvla-server/ and vla-inference/ with a single, model-agnostic
HTTP server. Supports SmolVLA, pi0.5 (stub), and future model backends.

Usage:
    uv run python server.py                       # defaults from config.yaml
    uv run python server.py --stub                # stub mode (no ML deps)
    VLA_DEVICE=cuda uv run python server.py       # CUDA override

HTTP API:
    GET  /health  -> {"status":"ok","model_loaded":true,"device":"mps"}
    GET  /config  -> {"action_dim":6,"cameras":["front"],"chunk_size":50}
    POST /predict -> {images, state, task} -> {actions, timestamp, inference_time_ms}
    POST /reset   -> {} -> {"ok":true}
"""

import argparse
import asyncio
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import yaml
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from models.base import VLAModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Config ───────────────────────────────────────────────────────

@dataclass
class ServerConfig:
    model: str = "smolvla"          # "smolvla" | "pi05" | "groot"
    model_path: str = "lerobot/smolvla_base"
    device: str = "mps"
    host: str = "0.0.0.0"
    port: int = 8000
    default_task: str = "Pick up the object."
    stub: bool = False
    # Stub action dimension (pi05 backend). None -> VLA_ACTION_DIM env var
    # -> default 6 (SO-101). Set 29 for Unitree G1 sim rollouts.
    action_dim: int | None = None
    # LoRA adapter (s3:// URI, local dir, or .tar.gz). Wraps base with PeftModel.
    adapter_path: str | None = None
    # Dataset stats.json for MEAN_STD un-normalization of actions.
    dataset_stats_path: str | None = None
    # Override camera feature names + empty camera padding.
    camera_names: list[str] | None = None
    empty_cameras: int | None = None
    # S3/RustFS credentials for s3:// paths.
    rustfs_endpoint: str | None = None
    rustfs_access_key: str | None = None
    rustfs_secret_key: str | None = None
    # Service token: require "Authorization: Bearer <auth_token>" on all
    # endpoints except /health. None disables auth (dev only).
    auth_token: str | None = None
    # GR00T N1.7 PolicyServer connection (model: groot).
    groot_host: str = "localhost"
    groot_port: int = 5555
    groot_api_token: str | None = None
    groot_video_key: str = "front"
    # Multi-camera list — wins over groot_video_key when set (e.g.
    # ["cam_left_high", "cam_right_high"]). None -> [groot_video_key].
    groot_video_keys: list[str] | None = None
    # Language key inside the GR00T observation; checkpoints trained with a
    # custom modality config may need "annotation.human.task_description".
    groot_language_key: str = "task"
    # Square resize edge for camera frames sent to GR00T. YAML null (None)
    # disables resizing and sends the native resolution.
    groot_image_size: int | None = 224
    groot_state_keys: dict[str, int] | None = None   # e.g. {"single_arm": 5, "gripper": 1}
    groot_action_keys: dict[str, int] | None = None  # e.g. {"single_arm": 5, "gripper": 1}
    groot_timeout_ms: int = 15000
    groot_ping_retries: int = 3  # startup ping attempts (2s apart) before giving up

    @classmethod
    def from_yaml(cls, path: str | Path = "config.yaml") -> "ServerConfig":
        path = Path(path)
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        return cls()


# ── Pydantic models ──────────────────────────────────────────────

class PredictRequest(BaseModel):
    image_b64: str | None = Field(
        None, description="Backward-compat: single base64-encoded JPEG (treated as front)"
    )
    images: dict[str, str] | None = Field(
        None, description="camera_name -> base64-encoded JPEG"
    )
    state: list[float] = Field(
        ..., description="Current joint positions"
    )
    task: str = Field(
        ..., description="Natural language instruction"
    )

    def resolved_images(self) -> dict[str, str]:
        """Return images dict, converting image_b64 to front if needed."""
        if self.images:
            return self.images
        if self.image_b64:
            return {"front": self.image_b64}
        return {}


class PredictResponse(BaseModel):
    actions: list[list[float]]
    timestamp: float
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: str = "ok"
    model: str = ""
    model_loaded: bool = False
    device: str = ""
    stub: bool = False
    auth_enabled: bool = False
    active_adapter_id: str | None = None


class LoadAdapterRequest(BaseModel):
    adapter_path: str = Field(
        ..., description="s3:// URI, absolute local path, or .tar.gz path"
    )
    adapter_id: str | None = Field(
        None, description="Caller-supplied identifier (e.g. modelVersionId)"
    )


class LoadAdapterResponse(BaseModel):
    adapter_id: str
    loaded_at: float
    load_time_ms: float
    info: dict


class ConfigResponse(BaseModel):
    action_dim: int
    chunk_size: int
    cameras: list[str]
    state_dim: int


# ── Model factory ────────────────────────────────────────────────

def create_model(config: ServerConfig) -> VLAModel:
    """Create the appropriate model backend."""
    if config.model == "groot":
        from models.groot import GR00TModel

        host = os.environ.get("VLA_HOST", config.groot_host)
        port = int(os.environ.get("VLA_ZMQ_PORT", str(config.groot_port)))
        stub = config.stub or os.environ.get("VLA_STUB", "").lower() in ("1", "true")
        api_token = os.environ.get("VLA_GROOT_API_TOKEN", config.groot_api_token)
        return GR00TModel(
            host=host,
            port=port,
            stub=stub,
            api_token=api_token,
            video_key=config.groot_video_key,
            video_keys=config.groot_video_keys,
            language_key=config.groot_language_key,
            image_size=config.groot_image_size,
            state_keys=config.groot_state_keys,
            action_keys=config.groot_action_keys,
            timeout_ms=config.groot_timeout_ms,
            ping_retries=config.groot_ping_retries,
        )

    if config.stub or config.model == "pi05":
        from models.pi05 import Pi05Model
        return Pi05Model(
            model_path=config.model_path,
            device=config.device,
            action_dim=config.action_dim,
        )

    if config.model == "smolvla":
        from models.smolvla import SmolVLAModel
        return SmolVLAModel(
            model_path=config.model_path,
            device=config.device,
            adapter_path=config.adapter_path,
            dataset_stats_path=config.dataset_stats_path,
            camera_names_override=config.camera_names,
            empty_cameras_override=config.empty_cameras,
            rustfs_endpoint=config.rustfs_endpoint,
            rustfs_access_key=config.rustfs_access_key,
            rustfs_secret_key=config.rustfs_secret_key,
        )

    raise ValueError(f"Unknown model: {config.model}")


# ── App ──────────────────────────────────────────────────────────

engine: VLAModel | None = None
config: ServerConfig | None = None
# Serializes /predict and /load-adapter so an in-flight inference cannot read
# a partially-mutated model during a hot-swap.
model_lock: asyncio.Lock = asyncio.Lock()


async def require_auth(authorization: str | None = Header(default=None)) -> None:
    """Service-token check for every endpoint except /health.

    Disabled when no auth_token is configured (a warning is logged at
    startup). /health stays open for load balancers and Docker healthchecks
    and never returns secrets.
    """
    expected = config.auth_token if config else None
    if not expected:
        return
    provided = ""
    if authorization and authorization.lower().startswith("bearer "):
        provided = authorization[7:].strip()
    if not provided:
        raise HTTPException(
            status_code=401,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # Compare as bytes: compare_digest raises TypeError on non-ASCII str,
    # which would turn a bad token into a 500 instead of a 403.
    if not secrets.compare_digest(provided.encode(), expected.encode()):
        raise HTTPException(status_code=403, detail="Invalid token")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, config
    config = app.state.config
    engine = create_model(config)

    logger.info("=" * 60)
    logger.info("VLA Inference Server")
    logger.info(f"  Model:  {config.model} ({config.model_path})")
    logger.info(f"  Device: {config.device}")
    logger.info(f"  Stub:   {config.stub}")
    logger.info(f"  Auth:   {'bearer token' if config.auth_token else 'DISABLED'}")
    logger.info(f"  Listen: {config.host}:{config.port}")
    logger.info("=" * 60)
    if not config.auth_token:
        logger.warning(
            "No auth_token configured — API is unauthenticated. "
            "Set auth_token in config.yaml or VLA_AUTH_TOKEN for production."
        )

    try:
        engine.load()
        logger.info("Model loaded. Server ready.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    yield
    logger.info("Shutting down.")
    engine.close()
    engine = None


app = FastAPI(
    title="VLA Inference Server",
    description="Consolidated VLA inference (SmolVLA, pi0.5, GR00T N1, ...)",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if engine and engine.is_loaded else "not_ready",
        model=config.model if config else "",
        model_loaded=engine is not None and engine.is_loaded,
        device=config.device if config else "",
        stub=engine.is_stub if engine else False,
        auth_enabled=bool(config.auth_token) if config else False,
        active_adapter_id=engine.active_adapter_id if engine else None,
    )


@app.get("/config", response_model=ConfigResponse, dependencies=[Depends(require_auth)])
async def get_config():
    if not engine or not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    model_info = engine.info()
    return ConfigResponse(
        action_dim=model_info.action_dim,
        chunk_size=model_info.chunk_size,
        cameras=model_info.cameras,
        state_dim=model_info.state_dim,
    )


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(require_auth)])
async def predict(request: PredictRequest):
    if not engine or not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # An empty state must never reach a backend — zero-padding it would
    # fabricate a robot pose and produce confident-but-wrong actions.
    if not request.state:
        raise HTTPException(
            status_code=422,
            detail="state must be a non-empty list of joint positions",
        )

    images = request.resolved_images()

    # Validate that all required cameras are present (extra cameras like wrist are OK)
    model_info = engine.info()
    expected = set(model_info.cameras)
    provided = set(images.keys())
    if not expected.issubset(provided):
        missing = expected - provided
        raise HTTPException(
            status_code=422,
            detail=f"Missing camera(s): {missing}. Expected: {expected}",
        )

    task = request.task or (config.default_task if config else "")

    async with model_lock:
        try:
            result = await asyncio.to_thread(
                engine.predict, images, request.state, task
            )
        except Exception as e:
            logger.error(f"Inference error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return PredictResponse(
        actions=result.actions,
        timestamp=time.time(),
        inference_time_ms=result.inference_time_ms,
    )


@app.post("/reset", dependencies=[Depends(require_auth)])
async def reset_policy():
    # Under model_lock and off the event loop: GR00T reset does a ZMQ
    # round-trip that must not interleave with an in-flight /predict on
    # the same (non-thread-safe) socket, nor block /health.
    if engine:
        async with model_lock:
            await asyncio.to_thread(engine.reset)
    return {"ok": True}


@app.post(
    "/load-adapter",
    response_model=LoadAdapterResponse,
    dependencies=[Depends(require_auth)],
)
async def load_adapter(request: LoadAdapterRequest):
    """Hot-swap a LoRA adapter onto the loaded base model.

    Body:
        adapter_path: s3:// URI, absolute local path, or .tar.gz path
        adapter_id: optional caller-supplied identifier (e.g. modelVersionId)

    Returns:
        adapter_id, loaded_at, load_time_ms, info dict
    """
    if not engine or not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    async with model_lock:
        t_start = time.perf_counter()
        try:
            result = await asyncio.to_thread(
                engine.load_adapter, request.adapter_path, request.adapter_id
            )
        except NotImplementedError as e:
            raise HTTPException(status_code=501, detail=str(e))
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=f"Adapter not found: {e}")
        except Exception as e:
            logger.error(f"Adapter load failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Adapter load failed: {e}")
        load_time_ms = (time.perf_counter() - t_start) * 1000

    return LoadAdapterResponse(
        adapter_id=result["adapter_id"],
        loaded_at=time.time(),
        load_time_ms=load_time_ms,
        info=result.get("info", {}),
    )


# ── CLI ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VLA Inference Server")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--stub", action="store_true", help="Stub mode (no ML deps)")
    args = parser.parse_args()

    cfg = ServerConfig.from_yaml(args.config)

    # Env overrides
    if env_device := os.environ.get("VLA_DEVICE"):
        cfg.device = env_device
    if env_model := os.environ.get("VLA_MODEL"):
        cfg.model = env_model
    if env_model_path := os.environ.get("VLA_MODEL_PATH"):
        cfg.model_path = env_model_path
    if env_port := os.environ.get("VLA_PORT"):
        cfg.port = int(env_port)
    if env_auth_token := os.environ.get("VLA_AUTH_TOKEN"):
        cfg.auth_token = env_auth_token
    if env_action_dim := os.environ.get("VLA_ACTION_DIM"):
        try:
            cfg.action_dim = int(env_action_dim)
        except ValueError:
            logger.warning(
                f"Invalid VLA_ACTION_DIM {env_action_dim!r}; "
                "keeping configured action_dim"
            )

    if args.stub:
        cfg.stub = True

    app.state.config = cfg

    import uvicorn
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
