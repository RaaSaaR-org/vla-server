# CLAUDE.md

VLA inference server for NeoDEM — extracted from `robot-management-system/vla-server/`.

## What This Is

Stateless FastAPI server that serves Vision-Language-Action model predictions over HTTP. Supports SmolVLA, pi0.5, and GR00T N1 backends.

## Key Files

- `server.py` — FastAPI entry point (endpoints: `/predict`, `/health`, `/config`, `/reset`)
- `models/` — model backends (smolvla.py, pi05.py, groot.py, base.py)
- `config.yaml.example` — default config template
- `pyproject.toml` — dependencies (base + optional smolvla/groot/dev groups)

## Dev Commands

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[smolvla,dev]"
uv run python server.py --stub    # no ML deps needed
pytest tests/
```

## Conventions

- Runs on port 8000 by default
- Config via `config.yaml` or env vars (`VLA_DEVICE`, `VLA_MODEL`, etc.)
- LeRobot must be installed from source for real inference (not on PyPI)
