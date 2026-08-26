"""
@file unifolm.py
@description Unitree UnifoLM-VLA backend for vla-server — HTTP bridge.

Viertes Modell der Kampagne. Wie GR00T und LingBot laeuft UnifoLM **nicht** in
diesem Prozess: es braucht torch 2.7.1+cu128, TensorFlow 2.15 und das
unifolm_vla-Paket, also ein eigenes venv. Die Bruecke dahinter ist
``vla-training/unifolm/serve/infer_server.py`` und spricht schlichtes HTTP.

Vertrag (identisch zu GR00T, pi0.5 und LingBot — sonst sind die Zahlen nicht
vergleichbar):
- ``state``: 43-dim float32 [rad]
  ``[L-Bein 6 | R-Bein 6 | Taille 3 | L-Arm 7 | R-Arm 7 | L-Hand 7 | R-Hand 7]``
- ``action``: **31-dim** ``[L-Arm 7 | R-Arm 7 | L-Hand 7 | R-Hand 7 | Taille 3]``
- eine Kamera ``ego_view``
- Aufgabentext ``"move the apple to the plate"`` (ohne Punkt)

**Die linke Hand wird hier NICHT dekodiert.** ``action[14:21]`` bleibt der
Griff-Code; der Dekoder sitzt flussabwaerts in
``vla-training/eval/run_apple_eval.py``. Doppelt dekodieren macht UnifoLM
gegen die anderen drei unlesbar.

**Chunk 25**, nicht 16 oder 50 — das ist ``NUM_ACTIONS_CHUNK`` der Embodiment
``g1_dex3_apple``. Der Eval-Runner schneidet mit ``--exec-horizon`` selbst ab,
genau wie bei den anderen Modellen.
"""

import logging
import time
from typing import Any

import requests

from .base import ModelConfig, PredictResult, VLAModel

logger = logging.getLogger(__name__)

ACTION_DIM = 31
STATE_DIM = 43
CHUNK_SIZE = 25
CAMERA_KEY = "ego_view"


class UnifolmModel(VLAModel):
    """Leitet /predict an die UnifoLM-Inferenzbruecke weiter."""

    def __init__(
        self,
        url: str = "http://127.0.0.1:8010",
        timeout_s: float = 60.0,
        connect_retries: int = 20,
        camera_key: str = CAMERA_KEY,
    ) -> None:
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s
        self.connect_retries = connect_retries
        self.camera_key = camera_key
        self._loaded = False
        self._chunk_size = CHUNK_SIZE
        self._session = requests.Session()

    def load(self) -> None:
        last = None
        for i in range(self.connect_retries):
            try:
                r = self._session.get(f"{self.url}/health", timeout=5)
                r.raise_for_status()
                h = r.json()
                if not h.get("ok"):
                    raise RuntimeError(f"Bruecke meldet nicht bereit: {h}")
                if h.get("action_dim") != ACTION_DIM or h.get("state_dim") != STATE_DIM:
                    raise RuntimeError(
                        f"Bruecke hat die falschen Dimensionen: {h} "
                        f"(erwartet {ACTION_DIM}/{STATE_DIM})"
                    )
                self._chunk_size = int(h.get("chunk_size", CHUNK_SIZE))
                self._loaded = True
                logger.info("UnifoLM-Bruecke bereit: %s", h)
                return
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(1.5)
        raise RuntimeError(
            f"UnifoLM-Bruecke unter {self.url} nicht erreichbar nach "
            f"{self.connect_retries} Versuchen: {last}"
        )

    def predict(self, images: dict[str, str], state: list[float], task: str) -> PredictResult:
        if not self._loaded:
            raise RuntimeError("Backend nicht geladen")
        if len(state) != STATE_DIM:
            raise ValueError(f"state hat {len(state)} Werte, erwartet {STATE_DIM}")
        if self.camera_key not in images:
            raise ValueError(
                f"Kamera {self.camera_key!r} fehlt; vorhanden: {sorted(images)}"
            )

        t0 = time.time()
        r = self._session.post(
            f"{self.url}/infer",
            json={"images": {self.camera_key: images[self.camera_key]},
                  "state": list(state), "task": task},
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        payload: dict[str, Any] = r.json()
        actions = payload["actions"]
        if not actions or len(actions[0]) != ACTION_DIM:
            raise RuntimeError(
                f"Bruecke lieferte Aktionsbreite {len(actions[0]) if actions else 0}, "
                f"erwartet {ACTION_DIM}"
            )
        return PredictResult(
            actions=actions,
            inference_time_ms=float(payload.get("inference_time_ms", (time.time() - t0) * 1000)),
        )

    def reset(self) -> None:
        # Die Bruecke ist zustandslos: jeder /infer-Aufruf sieht ein Bild und
        # einen Zustand. Nichts zurueckzusetzen.
        return

    def info(self) -> ModelConfig:
        return ModelConfig(
            action_dim=ACTION_DIM,
            chunk_size=self._chunk_size,
            cameras=[self.camera_key],
            state_dim=STATE_DIM,
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def close(self) -> None:
        self._session.close()
        self._loaded = False
