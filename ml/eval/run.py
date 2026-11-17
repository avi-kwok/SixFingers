"""make eval: metrics from cached features only -> docs/results.md (+ merged into the model's meta.json)."""
from __future__ import annotations

import argparse
import json
import re

import numpy as np

from ml.common.paths import REPO_ROOT, current_model_dir, data_dir
from ml.common.schemas import calibrated_logit, ensemble_score, load_meta, load_probe, save_meta
from ml.eval.metrics import summarize
from ml.features import store
from ml.features.augment import COMPRESSED_AUG

RESULTS = REPO_ROOT / "docs" / "results.md"
JUDGE_RE = re.compile(r"<!-- judge:start -->.*?<!-- judge:end -->", re.S)


def ensemble_on(model_dir, meta, df, aug):
    """Score rows of df that have features for every backbone. Returns (kept_df, ensemble, per_backbone scores)."""
    ids = set(df.image_id)
    per = {}
    for b in meta.backbones:
        d, X = store.load_matrix(df, b, aug)
        per[b] = (d, calibrated_logit(load_probe(model_dir / f"{b}.json"), X) if len(d) else np.array([]))
        ids &= set(d.image_id)
    kept = df[df.image_id.isin(ids)]
    logits = []
    for b in meta.backbones:
        d, lg = per[b]
        order = {i: k for k, i in enumerate(d.image_id)}
        logits.append(np.array([lg[order[i]] for i in kept.image_id]))
    return kept, (ensemble_score(logits) if len(kept) else np.array([])), {
        b: 1 / (1 + np.exp(-np.array([per[b][1][{i: k for k, i in enumerate(per[b][0].image_id)}[x]] for x in kept.image_id]))) for b in meta.backbones}


def baseline_scores():
    return {p.stem.removeprefix("baseline_"): json.loads(p.read_text()) for p in (data_dir() / "eval").glob("baseline_*.json")}


def fmt(v):
    return "n/a" if v is None else f"{v:.3f}"


def run(model_dir=None) -> dict:
    model_dir = (model_dir or current_model_dir()).resolve()
    meta = load_meta(model_dir / "meta.json")
    df = store.load_manifest()
    rt = df[df.split == "report_test"]
    res = {"model_version": meta.version, "variants": {}, "per_backbone": {}, "baselines": {}}
    for label, aug in (("clean", 0), ("compressed", COMPRESSED_AUG)):
        kept, s, per = ensemble_on(model_dir, meta, rt, aug)
        if len(kept):
            res["variants"][label] = summarize(kept.label, s, kept.generator, kept.source, meta.thresholds.t_ai)
            if label == "clean":
                res["per_backbone"] = {b: summarize(kept.label, v, kept.generator, kept.source) for b, v in per.items()}
        else:
            res["variants"][label] = {"n": 0, "note": "no cached features; run make features"}
    for name, sc in baseline_scores().items():
        d = rt[rt.image_id.isin(sc)]
        if len(d):
            res["baselines"][name] = summarize(d.label, [sc[i] for i in d.image_id], d.generator, d.source)
    lat = model_dir / "latency.json"
    res["latency_ms_cpu"] = json.loads(lat.read_text()) if lat.exists() else None
    meta.metrics["report_test"] = res
    save_meta(meta, model_dir / "meta.json")
    write_md(res, meta)
    return res


def write_md(res, meta):
    L = [f"# Results ({res['model_version']})", "",
         "Written by `make eval` from cached features. report_test holds out one whole generator plus RAISE reals.", "",
         f"Thresholds: t_ai={meta.thresholds.t_ai:.3f}, t_real={meta.thresholds.t_real:.3f}", "",
         "## Ensemble on report_test", "",
         "| variant | n | TPR@5%FPR | AUROC | ECE | TPR@t_ai | real FPR@t_ai | RAISE FPR@t_ai |", "|---|---|---|---|---|---|---|---|"]
    for k, m in res["variants"].items():
        L.append(f"| {k} | {m['n']} | {fmt(m.get('tpr_at_5fpr'))} | {fmt(m.get('auroc'))} | {fmt(m.get('ece'))} | "
                 f"{fmt(m.get('tpr_at_t_ai'))} | {fmt(m.get('real_fpr_at_t_ai'))} | {fmt(m.get('raise_real_fpr_at_t_ai'))} |")
    L += ["", "## Per-generator TPR at t_ai (clean)", ""]
    for g, v in res["variants"].get("clean", {}).get("per_generator_tpr", {}).items():
        L.append(f"- {g}: {v:.3f}")
    L += ["", "## Per backbone (clean)", "", "| backbone | TPR@5%FPR | AUROC |", "|---|---|---|"]
    for b, m in res["per_backbone"].items():
        L.append(f"| {b} | {fmt(m.get('tpr_at_5fpr'))} | {fmt(m.get('auroc'))} |")
    L += ["", "## Baseline (open HF detector, same report_test, clean)", ""]
    if res["baselines"]:
        L += ["| detector | n | TPR@5%FPR | AUROC |", "|---|---|---|---|"]
        L += [f"| {k} | {m['n']} | {fmt(m.get('tpr_at_5fpr'))} | {fmt(m.get('auroc'))} |" for k, m in res["baselines"].items()]
    else:
        L.append("Not run yet: `python -m ml.eval.baseline`.")
    L += ["", "## CPU latency per image (ms, batch 1)", ""]
    if res["latency_ms_cpu"]:
        L += [f"- {k}: {v}" for k, v in res["latency_ms_cpu"].items()]
    else:
        L.append("Not measured yet: `python -m ml.eval.latency`.")
    block = "\n<!-- judge:start -->\n## Judge comparison\n\nNot run yet.\n<!-- judge:end -->\n"
    if RESULTS.exists() and (m := JUDGE_RE.search(RESULTS.read_text())):
        block = "\n" + m.group(0) + "\n"
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text("\n".join(L) + "\n" + block)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None)
    from pathlib import Path
    a = ap.parse_args()
    run(Path(a.model_dir) if a.model_dir else None)
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
