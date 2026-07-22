# Changelog

All notable changes to the NeoDEM VLA server are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/). Versioning uses [CalVer](https://calver.org/) (`YYYY.MM.DD`) for daily releases.

## [v2026.07.22] - 2026-07-22

### Added

- apple-to-plate g1 config (NVIDIA GR00T-N1.7-AppleToPlate recipe) (#9)
- configurable action dimension for G1 (29-DoF) sim rollouts (#7)
- g1_dex3 support - multi-camera, configurable language key, native-resolution passthrough (#8)

### Maintenance

- move LeRobot to PyPI lerobot[smolvla,dataset]>=0.6.0 (#5)


## [v2026.07.04] - 2026-07-04

### Added

- **GR00T N1.7 inference backend** (`models/groot.py`) speaking the Isaac-GR00T N1.7 PolicyServer wire protocol: msgpack envelope, ZMQ `--api-token` auth, fail-fast startup ping, and automatic REQ-socket rebuild after timeouts (a single timeout no longer bricks the backend permanently). Modality keys configurable per embodiment, with SO-101 defaults.
- **Service-token auth** (`server.py`) — every endpoint except `/health` requires `Authorization: Bearer <token>` (constant-time compare, 401/403); `/health` reports `stub` and `auth_enabled` so clients can't mistake stub output for a real policy. Verified end-to-end against NVIDIA's unmodified `server_client.py` (57 unit/protocol tests, ZMQ-loopback PolicyServer emulator).

### Maintenance

- CalVer release automation — an always-open Release PR (`prepare-release-pr.yml`) that on merge tags `vYYYY.MM.DD`, publishes a GitHub Release, and builds/pushes `ghcr.io/raasaar-org/neodem-vla-server`.


## [v2026.04.12] - 2026-04-12

### Added

- Initial release of the VLA inference server, extracted from robot-management-system (TASK-150). Supports SmolVLA, pi0.5, and GR00T N1 backends via FastAPI.
