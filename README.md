# VLA Server

Consolidated VLA inference server for NeoDEM. Replaces the previous `smolvla-server/` and `vla-inference/` directories with a single FastAPI HTTP server that supports multiple model backends.

## Supported Models

| Model      | Status | Device       | Description                          |
|------------|--------|--------------|--------------------------------------|
| SmolVLA    | Ready  | MPS/CUDA/CPU | SmolVLA via LeRobot                  |
| pi0.5      | Stub   | CUDA         | pi0.5 via LeRobot (TASK-078)         |
| GR00T N1.7 | Ready  | ZMQ remote   | NVIDIA Isaac-GR00T PolicyServer via ZMQ (port 5555) |

## Setup

### Mac (Apple Silicon — MPS)

```bash
cd vla-server

# Create venv and install deps (pulls LeRobot ≥0.6.0 from PyPI)
uv venv && source .venv/bin/activate
uv pip install -e ".[smolvla]"

# Configure
cp config.yaml.example config.yaml
# Edit config.yaml: device: mps

# Run
uv run python server.py
```

### Linux (NVIDIA GPU — CUDA)

```bash
cd vla-server

uv venv && source .venv/bin/activate
uv pip install -e ".[smolvla]"   # pulls LeRobot ≥0.6.0 from PyPI

# Configure
cp config.yaml.example config.yaml
# Edit config.yaml: device: cuda

# Or use env override:
VLA_DEVICE=cuda uv run python server.py
```

### Stub Mode (No ML Dependencies)

For testing without torch/LeRobot — returns sine-wave actions:

```bash
cd vla-server
uv pip install -e .
uv run python server.py --stub
```

The stub defaults to 6-dim actions (SO-101). For other embodiments set the
action dimension via `action_dim` in `config.yaml` or `VLA_ACTION_DIM` —
e.g. a Unitree G1 (29 DoF) sim rollout:

```bash
VLA_ACTION_DIM=29 uv run python server.py --stub
```

## Authentication (Service Token)

Set a service token in `config.yaml` (`auth_token: "<token>"`) or via
`VLA_AUTH_TOKEN`. All endpoints except `/health` then require:

```
Authorization: Bearer <token>
```

```bash
# Generate a token
openssl rand -hex 32

# Run with auth
VLA_AUTH_TOKEN=$(openssl rand -hex 32) uv run python server.py

# Call with auth
curl -H "Authorization: Bearer $TOKEN" -X POST localhost:8000/predict -d '...'
```

Without a token the server runs open (dev mode) and logs a warning.
`/health` reports `auth_enabled` and `stub` so clients and monitoring can
verify they are talking to a real, secured policy — never point a robot at
a server whose `/health` says `"stub": true`.

The token authenticates but does not encrypt. Run the server inside a
private network (Tailscale mesh) or behind a TLS-terminating reverse proxy.

## HTTP API

| Method | Endpoint  | Auth | Description                                          |
|--------|-----------|------|------------------------------------------------------|
| GET    | /health   | open | `{"status":"ok","model_loaded":true,"stub":false,"auth_enabled":true}` |
| GET    | /config   | yes  | `{"action_dim":6,"cameras":["front"],"chunk_size":16}` |
| POST   | /predict  | yes  | `{images, state, task}` -> `{actions, timestamp, inference_time_ms}` |
| POST   | /reset    | yes  | Reset policy state between episodes                  |
| POST   | /load-adapter | yes | Hot-swap a LoRA adapter (SmolVLA)                 |

### /predict Request

```json
{
  "images": {"front": "<base64-encoded JPEG>"},
  "state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "task": "pick up the green object"
}
```

### /predict Response

```json
{
  "actions": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], ...],
  "timestamp": 1709000000.123,
  "inference_time_ms": 4.5
}
```

### GR00T N1.7 (NVIDIA Isaac-GR00T)

`VLA_MODEL=groot` connects to an Isaac-GR00T PolicyServer via ZMQ (port 5555).
Speaks the N1.7 wire protocol: msgpack envelope
(`{"endpoint", "data", "api_token"}`), nested observations
(`{"video", "state", "language"}` with batch/time dims), `get_action`
returning `(action, info)`.

#### Setup (GPU server side)

```bash
git clone https://github.com/NVIDIA/Isaac-GR00T.git && cd Isaac-GR00T
uv run python gr00t/eval/run_gr00t_server.py \
    --embodiment-tag NEW_EMBODIMENT \
    --model-path /path/to/so101-finetuned-checkpoint \
    --device cuda:0 --host 0.0.0.0 --port 5555
```

> Note: upstream `gr00t.eval.run_gr00t_server` has **no** `--api-token`
> flag. Leave `groot_api_token` empty unless you run a PolicyServer variant
> that enforces a token; run the ZMQ leg inside a private network instead.

Base model: [nvidia/GR00T-N1.7-3B](https://huggingface.co/nvidia/GR00T-N1.7-3B) —
fine-tune on your SO-101 dataset first; the modality keys
(`groot_video_key`, `groot_state_keys`, `groot_action_keys` in config.yaml)
must match the embodiment config of that checkpoint.

#### Setup (client side)

```bash
cd vla-server
uv pip install -e ".[groot]"
VLA_MODEL=groot VLA_HOST=<gpu-server-ip> VLA_GROOT_API_TOKEN=<zmq-token> \
    uv run python server.py
```

The server pings the PolicyServer at startup and fails fast if unreachable.
After a ZMQ timeout the connection is rebuilt automatically (REQ sockets
cannot be reused after an error).

#### Stub mode (no GPU)

```bash
VLA_MODEL=groot VLA_STUB=true uv run python server.py
```

### Unitree G1 EDU + Dex3-1 (embodiment `g1_dex3`)

Ready-made configs for GR00T-N1.7 checkpoints fine-tuned on the
`Unitree_G1_Dex3` dataset live in `configs/`:

| Config | Cameras | Checkpoint example |
|--------|---------|--------------------|
| `configs/g1_dex3_1cam.yaml` | `cam_right_high` | `n187_real_only_14k` |
| `configs/g1_dex3_2cam.yaml` | `cam_left_high` + `cam_right_high` | `n188_2cam_14k/checkpoint-8000` (⚠ checkpoint-10000 is truncated — don't auto-pick) |

Policy contract (from the checkpoint's `experiment_cfg`):

- **State**: `arms` (14) + `hands` (14) = 28-dim float32 joint **positions**
  in radians — no velocities. Arms = left shoulder pitch/roll/yaw, elbow,
  wrist roll/pitch/yaw, then right (same 7). Hands = Dex3 joints with a
  real left/right ordering asymmetry (see comments in the config files).
- **Action**: 28-dim **absolute** joint-position targets, same order,
  chunk length 16.
- **Camera frames**: native 480x640 RGB — `groot_image_size: null`
  disables the client-side 224x224 resize.
- **Language key**: `annotation.human.task_description` (the GR00T server
  asserts / KeyErrors on the default `task` key with these checkpoints).
- **Task instruction**: `"Put the bottle into the plate."` (exact string,
  trailing period).

Run it (the `--config` flag replaces the default `config.yaml`; all keys in
the file map 1:1 onto `ServerConfig` fields, unknown keys are ignored, and
`VLA_*` env variables still override afterwards):

```bash
# 1) GPU side — serve the checkpoint in WSL:
wsl.exe -d g1-eval -- bash -lc "bash /mnt/c/Unitree/_data/task185_serve_n17.sh n187_real_only_14k 6555"

# 2) HTTP side — this repo:
python server.py --config configs/g1_dex3_1cam.yaml
```

## Environment Variables

| Variable              | Description                     | Default          |
|-----------------------|---------------------------------|------------------|
| `VLA_DEVICE`          | Override device                 | from config.yaml |
| `VLA_MODEL`           | Override model backend          | from config.yaml |
| `VLA_MODEL_PATH`      | Override model path             | from config.yaml |
| `VLA_PORT`            | Override server port            | from config.yaml |
| `VLA_AUTH_TOKEN`      | HTTP service token              | from config.yaml |
| `VLA_HOST`            | GR00T ZMQ server host           | from config.yaml |
| `VLA_ZMQ_PORT`        | GR00T ZMQ server port           | from config.yaml |
| `VLA_GROOT_API_TOKEN` | GR00T PolicyServer API token    | from config.yaml |
| `VLA_STUB`            | Force stub mode (`true`/`1`)    | `false`          |
| `VLA_ACTION_DIM`      | Stub action dimension (pi05)    | `6` (SO-101); `29` for G1 |

## Tests

```bash
uv pip install -e ".[dev]"
pytest tests/
```
