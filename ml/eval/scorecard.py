"""Model scorecard: how often the detector catches each generator and how often it wrongly flags each real-photo source.

    python -m ml.eval.scorecard [--model-dir models/v15]

Reads cached features only (run make features first) and writes <model dir>/scorecard.json, which
GET /v1/scorecard serves. It never changes the model's other files (meta.json and the probes).

- Rates use the shipped threshold t_ai: an image counts as flagged when its ensemble score is >= t_ai.
- Generator rows come from report_test, whose generator the probes never saw.
- Real-source rows come from report_test (RAISE) and from the val split's real photos. Val rows are
  labelled "split": "val": the probes never trained on them, but t_ai was set so that at most 5% of
  val real photos score above it, so these rows sit near 5% by design.
- "clean" scores the images as stored; "compressed" scores the cached compressed copy of each image.
- The judge section reads the judge's report_test answers (data/eval/judge_cache_prod.jsonl) and the
  calibration file api/judge_calibration.json.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

from ml.common.paths import REPO_ROOT, current_model_dir, data_dir
from ml.common.schemas import load_meta
from ml.eval.metrics import summarize
from ml.eval.run import ensemble_on
from ml.features import store
from ml.features.augment import COMPRESSED_AUG

CALIBRATION = REPO_ROOT / "api" / "judge_calibration.json"


def _rows(kept, s, t_ai: float, label: int, col: str, split: str, rate: str) -> list[dict]:
    """One row per value of col among images with this label: n and the share scored >= t_ai."""
    s, is_label = np.asarray(s), (kept.label == label).to_numpy()
    out = []
    for name in sorted(set(kept[col][is_label])):
        sel = is_label & (kept[col] == name).to_numpy()
        out.append({col: name, "split": split, "n": int(sel.sum()), rate: float((s[sel] >= t_ai).mean())})
    return out


def judge_summary(cache: Path, calibration: Path, labels: dict) -> dict | None:
    """Judge accuracy on report_test. Correct = decisive and right, so 'uncertain' counts as wrong."""
    if not cache.exists() or not calibration.exists():
        return None
    cal = json.loads(calibration.read_text())
    answers = {}  # later lines win, as in ml.judge_eval's cache
    for line in cache.read_text().splitlines():
        r = json.loads(line)
        if r.get("arm") == cal["arm"] and r["image_id"] in labels:
            answers[r["image_id"]] = r.get("verdict")
    if not answers:
        return None
    right = [v == ("likely_ai" if labels[i] else "likely_real") for i, v in answers.items()]
    held = cal.get("report_test") or {}
    return {"model": cal["model"], "split": "report_test", "n": len(answers),
            "accuracy": float(np.mean(right)),
            "uncertain_rate": float(np.mean([v == "uncertain" for v in answers.values()])),
            "calibrated_ece": (held.get(cal["method"]) or {}).get("ece"), "calibration_method": cal["method"]}


def build(model_dir: Path, cache: Path | None = None, calibration: Path = CALIBRATION) -> dict:
    meta = load_meta(model_dir / "meta.json")
    t_ai = meta.thresholds.t_ai
    df = store.load_manifest()
    rt = df[df.split == "report_test"]
    val_real = df[(df.split == "val") & (df.label == 0)]
    variants = {}
    for name, aug in (("clean", 0), ("compressed", COMPRESSED_AUG)):
        kept, s, _ = ensemble_on(model_dir, meta, rt, aug)
        v = {"overall": None, "generators": [], "real_sources": []}
        if len(kept):
            m = summarize(kept.label, s, kept.generator, kept.source)
            v["overall"] = {"split": "report_test", **{k: m.get(k) for k in ("n", "auroc", "tpr_at_5fpr", "ece")}}
            v["generators"] = _rows(kept, s, t_ai, 1, "generator", "report_test", "catch_rate")
            v["real_sources"] = _rows(kept, s, t_ai, 0, "source", "report_test", "false_alarm_rate")
        kv, sv, _ = ensemble_on(model_dir, meta, val_real, aug)
        v["real_sources"] += _rows(kv, sv, t_ai, 0, "source", "val", "false_alarm_rate")
        variants[name] = v
    cache = cache or data_dir() / "eval" / "judge_cache_prod.jsonl"
    return {"model_version": meta.version, "date": date.today().isoformat(), "t_ai": t_ai,
            "detector": variants, "judge": judge_summary(cache, calibration, dict(zip(rt.image_id, rt.label)))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=None, help="model folder (default models/current)")
    a = ap.parse_args()
    model_dir = Path(a.model_dir or current_model_dir()).resolve()
    out = model_dir / "scorecard.json"
    out.write_text(json.dumps(build(model_dir), indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
