"""
@file pi05.py
@description pi0.5 (PI05) model backend for vla-server — real LeRobot policy.

Runs a fine-tuned ``PI05Policy`` **in-process** (no ZMQ/WebSocket hop like the
GR00T and LingBot backends): pi0.5 lives in LeRobot, so the policy, its
processor pipelines and the normalization stats all come out of one checkpoint
directory.

Modelled on ``models/smolvla.py`` (the other in-process LeRobot backend) —
same device handling, same base64-JPEG decoding, same PEFT adapter API. The
places where pi0.5 forces a different shape are marked with a WHY comment.

Checkpoint layout (``lerobot-train`` writes this, see
``lerobot/common/train_utils.py:131-141``)::

    <out>/checkpoints/<step>/pretrained_model/
        config.json                  # PI05Config incl. input/output features
        model.safetensors            # full finetune   … OR …
        adapter_config.json          # PEFT/LoRA finetune (scripts/32_train_pi05_lora.sh)
        adapter_model.safetensors
        policy_preprocessor.json     # + *_step_*.safetensors: QUANTILE stats
        policy_postprocessor.json

Both variants are supported; the PEFT case is detected by ``adapter_config.json``
and resolved exactly like ``lerobot/policies/factory.py:565-588``.

Contract for our apple-pick-and-place finetune (``_data/apple_pnp/CONTRACT.md``):
- ``observation.state``: 43-dim float32 [rad]
  ``[left_leg 6 | right_leg 6 | waist 3 | left_arm 7 | right_arm 7 | left_hand 7 | right_hand 7]``
- ``action``: **31-dim** ``[left_arm 7 | right_arm 7 | left_hand 7 | right_hand 7 | waist 3]``.
  43-dim actions are impossible for pi0.5 (``PI05Config.max_action_dim`` is 32),
  hence the derived dataset built by
  ``vla-training/scripts/33_make_dataset_action31.py`` (``dataset_v30_a31``).
- one camera ``observation.images.ego_view``, NATIVE 640x480 RGB — no client
  resize; ``modeling_pi05.py:1190-1191`` letterboxes to 224x224 itself.
- task string: ``"move the apple to the plate"``
- normalization: QUANTILES for state *and* action
  (``configuration_pi05.py:72-78``) — the stats are baked into the saved
  processor pipelines, so they must be loaded from the checkpoint, never
  re-created from scratch.

**Chunk sizes.** ``PI05Config.chunk_size``/``n_action_steps`` are 50, and this
backend returns the full 50-step chunk from ``predict_action_chunk()``. The
"chunk 16 / exec-horizon 8" in CONTRACT.md is a *client* policy: the eval
runner slices ``[:args.exec_horizon]`` off the response
(``vla-training/eval/run_apple_eval.py:219``), exactly as it does for the
16-step GR00T chunk. Set ``pi05_max_chunk_size: 16`` to truncate server-side
for strict parity with GR00T.

**The left-hand action block is a grip CODE, not radians** (CONTRACT.md:31-40).
It is deliberately NOT decoded here: ``models/groot.py`` returns the raw 31-dim
policy output too, and the decode happens downstream in the eval runner
(``vla-training/eval/run_apple_eval.py:221`` via
``eval/hand_grip_decoder.py::decode_left_hand``) and in the Isaac bridge
(``eval/isaac_dds_bridge.py:439``). Decoding here would double-decode those
paths and make the two backends non-comparable.

Torch/LeRobot are imported lazily inside the methods (unlike ``smolvla.py``,
which imports torch at module level) so that merely importing this module stays
cheap and fails with the same "install the extras" message the GR00T
(``groot.py:110-114``) and LingBot (``lingbot.py:203-210``) backends give.
"""

import base64
import io
import json
import logging
import re
import shutil
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
from PIL import Image

from .base import ModelConfig, PredictResult, VLAModel

logger = logging.getLogger(__name__)

# g1_apple_pnp defaults — only used until the real config.json is read in
# load(); the checkpoint always wins.
ACTION_DIM = 31
CHUNK_SIZE = 50  # PI05Config.chunk_size
STATE_DIM = 43
CAMERA_KEY = "ego_view"

# LeRobot feature keys (kept as literals like the other backends, so this file
# does not need lerobot importable just to be read).
OBS_STATE_KEY = "observation.state"
OBS_IMAGES_PREFIX = "observation.images."
ACTION_KEY = "action"
TASK_KEY = "task"

# Written by PolicyProcessorPipeline.save_pretrained()
# (lerobot/utils/constants.py:57-58).
PREPROCESSOR_FILE = "policy_preprocessor.json"
POSTPROCESSOR_FILE = "policy_postprocessor.json"

_HF_REPO_RE = re.compile(r"^[\w.-]+/[\w.-]+$")


class Pi05Model(VLAModel):
    """pi0.5 via the LeRobot ``PI05Policy`` API, running in this process.

    Supports full-finetune and PEFT/LoRA checkpoints, plus LoRA hot-swap
    through ``/load-adapter``.
    """

    def __init__(
        self,
        model_path: str = "",
        device: str = "cuda",
        adapter_path: str | None = None,
        dataset_stats_path: str | None = None,
        camera_key: str = CAMERA_KEY,
        image_size: int | None = None,
        state_dim: int = STATE_DIM,
        max_chunk_size: int | None = None,
        rustfs_endpoint: str | None = None,
        rustfs_access_key: str | None = None,
        rustfs_secret_key: str | None = None,
    ):
        self.model_path = model_path
        self.device = device
        self.adapter_path = adapter_path
        self._dataset_stats_path = dataset_stats_path
        # Camera name as sent by /predict clients. The policy key is
        # "observation.images.<camera_key>"; the real list is re-read from the
        # checkpoint config in load().
        self.camera_key = camera_key
        # Square resize edge; None = send the native 640x480 through, which is
        # what CONTRACT.md:47 demands (pi0.5 letterboxes to 224 internally).
        self.image_size = image_size
        # Truncate the returned chunk (None = return all chunk_size steps).
        self.max_chunk_size = max_chunk_size
        self._rustfs_endpoint = rustfs_endpoint
        self._rustfs_access_key = rustfs_access_key
        self._rustfs_secret_key = rustfs_secret_key

        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        # The PI05Config is kept explicitly instead of reading policy.config
        # (as smolvla.py:198-201 does): once wrapped in a PeftModel, `.config`
        # resolves to PEFT's own config attribute, not PI05Config.
        self._cfg = None

        self._action_dim = ACTION_DIM
        self._chunk_size = CHUNK_SIZE
        self._state_dim = state_dim
        self._cameras: list[str] = [camera_key]

        self._adapter_scratch_dir: Path | None = None
        self._active_adapter_id: str | None = None
        self._loaded_adapter_names: set[str] = set()
        self._warned_partial_state = False

    # ── Lifecycle ────────────────────────────────────────────────

    def load(self) -> None:
        if self.policy is not None:
            logger.warning("Pi05Model already loaded, skipping")
            return

        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError(
                "pi0.5 backend requires torch + lerobot: "
                "uv pip install -e '.[smolvla]'  (same extras as SmolVLA)"
            )

        ckpt = self._resolve_checkpoint(self.model_path)
        logger.info(f"Loading pi0.5 from {ckpt} on {self.device}")

        PI05Policy = self._import_pi05_policy()
        PreTrainedConfig = self._import_pretrained_config()
        make_pre_post_processors = self._import_processor_factory()

        # 1. Policy config. Read first so we can pin the device before the
        #    3B model is materialised — PI05Policy.__init__ already does
        #    `self.model.to(config.device)` (modeling_pi05.py:939).
        cfg = PreTrainedConfig.from_pretrained(pretrained_name_or_path=str(ckpt))
        cfg.device = self.device
        self._cfg = cfg

        # 2. Weights. A PEFT checkpoint only carries the adapter; the base
        #    model id lives in adapter_config.json
        #    (factory.py:565-588 does the same dance).
        peft_dir = self._peft_dir(ckpt)
        if peft_dir is not None:
            from peft import PeftConfig, PeftModel

            peft_config = PeftConfig.from_pretrained(str(peft_dir))
            base = peft_config.base_model_name_or_path
            if not base:
                raise RuntimeError(
                    f"{peft_dir / 'adapter_config.json'} has no "
                    "base_model_name_or_path — cannot tell which pi0.5 base "
                    "this adapter belongs to"
                )
            logger.info(f"PEFT checkpoint detected; base model: {base}")
            self.policy = PI05Policy.from_pretrained(base, config=cfg)
            self.policy = PeftModel.from_pretrained(
                self.policy, str(peft_dir), config=peft_config
            )
            self._loaded_adapter_names.add("default")
            self._active_adapter_id = "default"
        else:
            self.policy = PI05Policy.from_pretrained(str(ckpt), config=cfg)

        # 3. Optional extra LoRA adapter on top (config: adapter_path).
        if self.adapter_path:
            self._wrap_with_adapter()
            self._loaded_adapter_names.add("default")
            self._active_adapter_id = "default"

        # 4. Processor pipelines. These carry the QUANTILE stats and the
        #    PaliGemma tokenizer, and pi0.5 is unusable without them — unlike
        #    smolvla.py, which tokenizes by hand and un-normalizes with a
        #    stats JSON. Loading them from the checkpoint is the documented
        #    path (async_inference/policy_server.py:157-165).
        overrides_pre: dict[str, Any] = {"device_processor": {"device": self.device}}
        overrides_post: dict[str, Any] = {"device_processor": {"device": "cpu"}}
        if self._dataset_stats_path:
            stats = self._load_dataset_stats()
            overrides_pre["normalizer_processor"] = {"stats": stats}
            overrides_post["unnormalizer_processor"] = {"stats": stats}

        proc_src = self._processor_source(ckpt)
        try:
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                cfg,
                pretrained_path=proc_src,
                preprocessor_overrides=overrides_pre,
                postprocessor_overrides=overrides_post,
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not load the pi0.5 processor pipelines from {proc_src}: {e}. "
                f"The checkpoint must contain {PREPROCESSOR_FILE} and "
                f"{POSTPROCESSOR_FILE} — they hold the QUANTILE normalization "
                "stats, without which the policy outputs garbage."
            ) from e

        self.policy.to(self.device)
        self.policy.eval()

        self._read_dims(cfg)

        logger.info(
            f"pi0.5 loaded: action_dim={self._action_dim}, "
            f"chunk_size={self._chunk_size}, state_dim={self._state_dim}, "
            f"cameras={self._cameras}"
        )

    def reset(self) -> None:
        # Clears PI05Policy._action_queue (modeling_pi05.py:1126-1131). Only
        # select_action() uses that queue, but resetting keeps episode
        # boundaries clean if the policy ever grows more state.
        if self.policy is not None and hasattr(self.policy, "reset"):
            self.policy.reset()

    def close(self) -> None:
        """Release the policy and any adapter scratch space. Idempotent."""
        if self.policy is None:
            return
        logger.info("Unloading pi0.5")
        import gc

        import torch

        del self.policy
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        if self._adapter_scratch_dir and self._adapter_scratch_dir.exists():
            shutil.rmtree(self._adapter_scratch_dir, ignore_errors=True)
        self._adapter_scratch_dir = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # torch.mps.empty_cache existiert in JEDEM Torch-Build, auch in den
        # cu128-Wheels auf dz-226 — hasattr() ist deshalb kein brauchbarer
        # Waechter und der Aufruf wirft dort "Cannot execute emptyCache()
        # without MPS backend". Das passiert in close(), also beim
        # Herunterfahren: die Ausnahme ueberdeckte den echten Abbruchgrund
        # (gemessen 2026-08-03: ein Bind-Konflikt war nur noch als
        # MPS-RuntimeError sichtbar). Nach der Verfuegbarkeit fragen, nicht
        # nach dem Attribut.
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

    def info(self) -> ModelConfig:
        chunk = self._chunk_size
        if self.max_chunk_size is not None:
            chunk = min(chunk, self.max_chunk_size)
        return ModelConfig(
            action_dim=self._action_dim,
            chunk_size=chunk,
            cameras=list(self._cameras),
            state_dim=self._state_dim,
        )

    @property
    def is_loaded(self) -> bool:
        return self.policy is not None

    @property
    def is_stub(self) -> bool:
        return False  # real policy — no sine waves left in this backend

    @property
    def active_adapter_id(self) -> str | None:
        return self._active_adapter_id

    # ── Inference ────────────────────────────────────────────────

    def predict(
        self,
        images: dict[str, str],
        state: list[float],
        task: str,
    ) -> PredictResult:
        if self.policy is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        import torch

        t_start = time.perf_counter()
        batch = self._build_observation(images, state, task)

        with torch.no_grad():
            # WHY predict_action_chunk and not select_action (smolvla.py:163):
            # select_action pops ONE action off an internal queue, but our
            # /predict contract returns a whole chunk that the client
            # executes open-loop (exec-horizon 8). Going through the queue
            # would silently replay stale actions across HTTP calls.
            processed = self.preprocessor(batch)
            chunk = self.policy.predict_action_chunk(processed)
            if chunk.ndim == 2:  # (chunk, dim) -> (1, chunk, dim)
                chunk = chunk.unsqueeze(0)

            # The postprocessor (un-normalize + relative->absolute) works on
            # one (B, action_dim) step at a time, so the chunk is fed through
            # step by step — same loop as policy_server.py:370-379.
            steps = [self.postprocessor(chunk[:, i, :]) for i in range(chunk.shape[1])]
            actions = torch.stack(steps, dim=1).squeeze(0)

        action_np = actions.detach().to("cpu", torch.float32).numpy()
        if action_np.ndim == 1:
            action_np = action_np.reshape(1, -1)
        if self.max_chunk_size is not None:
            action_np = action_np[: self.max_chunk_size]

        # NOTE: rows are the raw 31-dim policy output. action[14:21]
        # (left_hand) is a grip CODE — the caller decodes it, see module
        # docstring.
        inference_time_ms = (time.perf_counter() - t_start) * 1000
        return PredictResult(
            actions=action_np.tolist(), inference_time_ms=inference_time_ms
        )

    # ── Observation building ─────────────────────────────────────

    def _build_observation(
        self, images: dict[str, str], state: list[float], task: str
    ) -> dict:
        """Convert the /predict payload into an UNBATCHED LeRobot batch dict.

        Unbatched on purpose: the pi0.5 preprocessor starts with
        ``AddBatchDimensionProcessorStep`` (processor_pi05.py:141), which
        unsqueezes 1-D states / 3-D images and wraps the task string in a
        list. Adding a batch dim here (as smolvla.py:445 does for its
        hand-built observation) would produce 5-D image tensors.

        Tensors stay on the CPU as well — ``DeviceProcessorStep`` at the end
        of the pipeline moves them, and the state discretizer in between calls
        ``.cpu()`` anyway (processor_pi05.py:76).
        """
        import torch

        batch: dict[str, Any] = {}

        for cam in self._cameras:
            if cam not in images:
                raise ValueError(
                    f"missing camera '{cam}' in request (got {sorted(images)})"
                )
            img = Image.open(io.BytesIO(base64.b64decode(images[cam]))).convert("RGB")
            if self.image_size is not None:
                img = img.resize((self.image_size, self.image_size))
            arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC in [0, 1]
            batch[f"{OBS_IMAGES_PREFIX}{cam}"] = torch.from_numpy(arr).permute(2, 0, 1)

        # Never fabricate a robot pose: an empty state must fail loudly
        # instead of silently becoming an all-zero joint configuration.
        if not state:
            raise ValueError(
                "state must be a non-empty list of joint positions "
                f"(expected {self._state_dim} values)"
            )
        if len(state) != self._state_dim and not self._warned_partial_state:
            self._warned_partial_state = True
            logger.warning(
                f"state has {len(state)} values, expected {self._state_dim}; "
                f"padding/truncating (logged once)"
            )
        padded = (list(state) + [0.0] * self._state_dim)[: self._state_dim]
        batch[OBS_STATE_KEY] = torch.tensor(padded, dtype=torch.float32)

        # Goes into COMPLEMENTARY_DATA under "task"
        # (processor/converters.py:156-167) and ends up in the PaliGemma
        # prompt built by Pi05PrepareStateTokenizerProcessorStep.
        batch[TASK_KEY] = task
        return batch

    # ── Checkpoint resolution ────────────────────────────────────

    def _resolve_checkpoint(self, path: str) -> str | Path:
        """Validate the checkpoint location before a 3B model is built."""
        if not path:
            raise RuntimeError(
                "pi0.5 backend needs a checkpoint: set model_path to a "
                "'.../checkpoints/<step>/pretrained_model' directory or a "
                "HuggingFace repo id (e.g. lerobot/pi05_base)"
            )
        p = Path(path)
        if p.exists():
            if not p.is_dir():
                raise FileNotFoundError(
                    f"pi0.5 model_path is not a directory: {p}"
                )
            if not (p / "config.json").exists():
                raise FileNotFoundError(
                    f"No config.json in {p} — this is not a LeRobot "
                    "pretrained_model directory"
                )
            has_weights = (
                (p / "model.safetensors").exists()
                or (p / "model.safetensors.index.json").exists()
                or (p / "adapter_config.json").exists()
            )
            if not has_weights:
                # PI05Policy.from_pretrained only PRINTS on a failed state-dict
                # load and returns a randomly initialised model
                # (modeling_pi05.py:1007-1010, 1056-1058), so the missing-file
                # case has to be caught here or it becomes silent nonsense.
                raise FileNotFoundError(
                    f"No model.safetensors and no adapter_config.json in {p} — "
                    "checkpoint is incomplete"
                )
            return p
        if _HF_REPO_RE.match(path):
            return path  # HuggingFace repo id — resolved by from_pretrained
        raise FileNotFoundError(f"pi0.5 checkpoint not found: {path}")

    def _peft_dir(self, ckpt: str | Path) -> Path | None:
        """Return the checkpoint dir if it is a PEFT adapter, else None."""
        if isinstance(ckpt, Path) and (ckpt / "adapter_config.json").exists():
            return ckpt
        # OFFEN: a PEFT checkpoint pushed to the Hub is not detected here — we
        # would have to download adapter_config.json first. Conservative
        # choice: treat hub ids as full checkpoints (which lerobot/pi05_base
        # and every merged finetune are).
        return None

    def _processor_source(self, ckpt: str | Path) -> str:
        """Where policy_pre/postprocessor.json live.

        Always the checkpoint directory itself: save_checkpoint() writes the
        pipelines next to the (adapter) weights, for PEFT runs too
        (common/train_utils.py:131-141).
        """
        if isinstance(ckpt, Path) and not (ckpt / PREPROCESSOR_FILE).exists():
            raise FileNotFoundError(
                f"{ckpt / PREPROCESSOR_FILE} is missing — the checkpoint was "
                "saved without its processor pipelines, so the QUANTILE "
                "normalization stats are unavailable"
            )
        return str(ckpt)

    def _read_dims(self, cfg) -> None:
        """Read the real dims/cameras out of the checkpoint config."""
        try:
            out = getattr(cfg, "output_features", {}) or {}
            if ACTION_KEY in out:
                self._action_dim = int(out[ACTION_KEY].shape[0])
            inp = getattr(cfg, "input_features", {}) or {}
            if OBS_STATE_KEY in inp:
                self._state_dim = int(inp[OBS_STATE_KEY].shape[0])
            self._chunk_size = int(getattr(cfg, "chunk_size", CHUNK_SIZE))
            cams = [
                k[len(OBS_IMAGES_PREFIX) :]
                for k in (getattr(cfg, "image_features", None) or {})
                if k.startswith(OBS_IMAGES_PREFIX)
            ]
            if cams:
                self._cameras = cams
        except Exception:
            logger.warning(
                "Could not read pi0.5 config dims, using g1_apple_pnp defaults"
            )

        if self._action_dim != ACTION_DIM:
            logger.warning(
                f"pi0.5 checkpoint has action_dim={self._action_dim}, but the "
                f"g1_apple_pnp contract is {ACTION_DIM} "
                f"[L-arm 7 | R-arm 7 | L-hand 7 | R-hand 7 | waist 3]"
            )

    # ── LeRobot imports (version-tolerant, mirrors smolvla.py:87-102) ──

    def _import_pi05_policy(self):
        try:
            from lerobot.policies.pi05.modeling_pi05 import PI05Policy

            return PI05Policy
        except (ImportError, AttributeError):
            try:
                from lerobot.common.policies.pi05.modeling_pi05 import PI05Policy

                return PI05Policy
            except (ImportError, AttributeError) as e:
                raise RuntimeError(
                    "Could not import PI05Policy. Tried lerobot.policies and "
                    f"lerobot.common.policies. Check the LeRobot install. Error: {e}"
                ) from e

    def _import_pretrained_config(self):
        try:
            from lerobot.configs.policies import PreTrainedConfig
        except (ImportError, AttributeError):
            from lerobot.common.configs.policies import PreTrainedConfig
        # Importing the pi05 config registers the "pi05" choice with draccus,
        # without which PreTrainedConfig.from_pretrained cannot decode
        # config.json (configuration_pi05.py:28).
        try:
            from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: F401
        except (ImportError, AttributeError):
            from lerobot.common.policies.pi05.configuration_pi05 import (  # noqa: F401
                PI05Config,
            )
        return PreTrainedConfig

    def _import_processor_factory(self):
        try:
            from lerobot.policies.factory import make_pre_post_processors

            return make_pre_post_processors
        except (ImportError, AttributeError):
            try:
                from lerobot.common.policies.factory import make_pre_post_processors

                return make_pre_post_processors
            except (ImportError, AttributeError) as e:
                raise RuntimeError(
                    "Could not import make_pre_post_processors from LeRobot. "
                    f"Error: {e}"
                ) from e

    # ── Dataset stats override ───────────────────────────────────

    def _load_dataset_stats(self) -> dict:
        """Load a stats JSON to override the checkpoint's normalization stats.

        Only needed when the stats in the checkpoint are wrong/absent — pi0.5
        uses QUANTILES, so the JSON must carry q01/q99 per feature, not the
        mean/std that smolvla.py:411-416 reads.
        """
        path = self._dataset_stats_path
        if path.startswith("s3://"):
            import boto3
            from botocore.client import Config as BotoConfig

            parsed = urlparse(path)
            client = boto3.client(
                "s3",
                endpoint_url=self._rustfs_endpoint,
                aws_access_key_id=self._rustfs_access_key,
                aws_secret_access_key=self._rustfs_secret_key,
                config=BotoConfig(signature_version="s3v4"),
                region_name="us-east-1",
            )
            raw = client.get_object(
                Bucket=parsed.netloc, Key=parsed.path.lstrip("/")
            )["Body"].read()
            stats = json.loads(raw)
        else:
            stats = json.loads(Path(path).read_text())
        logger.info(f"Dataset stats override loaded from {path}: {sorted(stats)}")
        return stats

    # ── Adapter loading (same API + strategies as smolvla.py) ────

    def load_adapter(self, adapter_path: str, adapter_id: str | None = None) -> dict:
        """Hot-swap a LoRA adapter onto the loaded pi0.5 base model."""
        if self.policy is None:
            raise RuntimeError("Base model not loaded; cannot apply adapter")

        from peft import PeftModel

        adapter_dir = self._resolve_adapter_dir(adapter_path)
        name = adapter_id or f"adapter-{int(time.time() * 1000)}"

        if isinstance(self.policy, PeftModel):
            if name in self._loaded_adapter_names:
                logger.info(f"Activating already-loaded adapter '{name}'")
                self.policy.set_adapter(name)
            else:
                logger.info(f"Loading adapter '{name}' from {adapter_dir}")
                self.policy.load_adapter(str(adapter_dir), adapter_name=name)
                self.policy.set_adapter(name)
                self._loaded_adapter_names.add(name)
            strategy = "peft_set_adapter"
        else:
            logger.info(
                f"Wrapping base policy with adapter '{name}' from {adapter_dir}"
            )
            self.policy = PeftModel.from_pretrained(
                self.policy, str(adapter_dir), adapter_name=name
            )
            self.policy.to(self.device)
            self.policy.eval()
            self._loaded_adapter_names.add(name)
            strategy = "peft_from_pretrained"

        self._active_adapter_id = name
        logger.info(f"Adapter '{name}' active (strategy={strategy})")
        return {
            "adapter_id": name,
            "info": {
                "strategy": strategy,
                "path": adapter_path,
                "total_loaded": len(self._loaded_adapter_names),
            },
        }

    def _wrap_with_adapter(self) -> None:
        from peft import PeftModel

        adapter_dir = self._resolve_adapter_dir(self.adapter_path)
        logger.info(f"Applying LoRA adapter from {adapter_dir}")
        self.policy = PeftModel.from_pretrained(self.policy, str(adapter_dir))
        logger.info("LoRA adapter applied successfully")

    def _resolve_adapter_dir(self, path: str) -> Path:
        if path.startswith("s3://"):
            return self._download_and_unpack_s3(path)
        p = Path(path)
        if p.is_dir():
            return self._find_adapter_dir(p)
        if p.is_file() and p.suffix in (".gz", ".tar", ".tgz"):
            return self._unpack_tarball(p)
        raise FileNotFoundError(f"Adapter path not found: {path}")

    def _download_and_unpack_s3(self, s3_uri: str) -> Path:
        import boto3
        from botocore.client import Config as BotoConfig

        parsed = urlparse(s3_uri)
        bucket, key = parsed.netloc, parsed.path.lstrip("/")
        if not self._rustfs_endpoint:
            raise RuntimeError(
                "adapter_path is s3:// but rustfs_endpoint not configured"
            )
        logger.info(f"Downloading adapter from {s3_uri}")
        client = boto3.client(
            "s3",
            endpoint_url=self._rustfs_endpoint,
            aws_access_key_id=self._rustfs_access_key,
            aws_secret_access_key=self._rustfs_secret_key,
            config=BotoConfig(signature_version="s3v4"),
            region_name="us-east-1",
        )
        scratch = Path(tempfile.mkdtemp(prefix="pi05-adapter-"))
        self._adapter_scratch_dir = scratch
        tar_path = scratch / "adapter.tar.gz"
        client.download_file(bucket, key, str(tar_path))
        logger.info(f"Downloaded {tar_path.stat().st_size} bytes")
        return self._unpack_tarball(tar_path, dest=scratch / "unpacked")

    def _unpack_tarball(self, tar_path: Path, dest: Path | None = None) -> Path:
        if dest is None:
            dest = Path(tempfile.mkdtemp(prefix="pi05-adapter-"))
            self._adapter_scratch_dir = dest
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r:*") as tf:
            tf.extractall(dest)
        return self._find_adapter_dir(dest)

    def _find_adapter_dir(self, root: Path) -> Path:
        if (root / "adapter_config.json").exists():
            return root
        for sub in root.iterdir():
            if sub.is_dir() and (sub / "adapter_config.json").exists():
                return sub
        raise FileNotFoundError(f"No adapter_config.json under {root}")
