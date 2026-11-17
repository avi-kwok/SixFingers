"""Step 4: Claude reads the image plus provenance findings (never detector scores) and returns a structured verdict."""
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import anthropic

from ml.common.api_models import JudgeOutput, Provenance
from ml.common.judge_backend import BackendError

log = logging.getLogger("sixfingers")

MODEL = "claude-haiku-5-5"
TIMEOUT_S = 15.0
SCHEMA = anthropic.transform_schema(JudgeOutput.model_json_schema())

SYSTEM = """You judge whether a still image is fully AI-generated. You get the image plus provenance findings. \
A signed C2PA AI manifest is strong evidence for AI; a missing manifest means nothing. Look for concrete visual cues \
(hands, text, lighting, textures, reflections, symmetry). Keep explanation under 80 words. \
Edited or partly AI images are out of scope: say uncertain."""


class JudgeUnavailable(Exception):
    def __init__(self, reason: str):
        self.reason = reason


class Budget:
    """Daily call counter persisted to a JSON file so restarts do not reset it."""

    def __init__(self, path: Path, limit: int):
        self.path, self.limit, self._lock = path, limit, threading.Lock()

    def take(self) -> bool:
        today = datetime.now(timezone.utc).date().isoformat()
        with self._lock:
            try:
                st = json.loads(self.path.read_text())
            except Exception:
                st = {}
            if st.get("date") != today:
                st = {"date": today, "count": 0}
            if st["count"] >= self.limit:
                return False
            st["count"] += 1
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(st))
            os.replace(tmp, self.path)
            return True


def _evidence(prov: Provenance) -> str:
    return "Provenance JSON:\n" + json.dumps(prov.model_dump()) + "\nIs this image AI-generated?"


class Judge:
    def __init__(self, backend, budget: Budget, enabled: bool = True):
        self.backend, self.budget, self.enabled = backend, budget, enabled and backend is not None

    def judge(self, data: bytes, mime: str, prov: Provenance) -> JudgeOutput:
        if not self.enabled:
            raise JudgeUnavailable("disabled")
        if not self.budget.take():
            raise JudgeUnavailable("budget")
        try:
            out, _usage = self.backend.ask(
                model=MODEL, system=SYSTEM, image=data, evidence=_evidence(prov),
                schema=SCHEMA, max_tokens=800, timeout=TIMEOUT_S)
        except BackendError as e:
            log.warning("judge unavailable: %s", e.reason)
            raise JudgeUnavailable(e.reason if e.reason in ("timeout", "refusal") else "error") from None
        except Exception as e:  # any backend failure must still yield detector_only
            log.warning("judge failed: %s", type(e).__name__)
            raise JudgeUnavailable("error") from None
        return out
