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
    --device cuda:0 --host 0.0.0.0 --port 5555 \
    --api-token "<zmq-token>"
```

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

## Tests

```bash
uv pip install -e ".[dev]"
pytest tests/
```
