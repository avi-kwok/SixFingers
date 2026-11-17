"""Judge comparison on a stratified report_test slice. HARD spend cap, cached, resume-safe.

    python -m ml.judge_eval.run --approve-spend 10 [--n 300]

The judge backend comes from SIXFINGERS_JUDGE_BACKEND: api (default, needs ANTHROPIC_API_KEY) or command
(runs the program in SIXFINGERS_JUDGE_CMD). When that program exits 75 the run waits its retry_after_s
(default 1800 s) and resumes from the cache.
Spend is the sum of cached per-call costs (usage tokens x list price), so a re-run never pays twice.
Before each call the worst-case cost (estimated input + max_tokens of output) must still fit under the cap.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd

from ml.common.api_models import JudgeOutput
from ml.common.judge_backend import DEFAULT_RETRY_S, ApiBackend, BackendError, from_env
from ml.common.paths import current_model_dir, data_dir
from ml.common.schemas import band_for, load_meta
from ml.eval.run import JUDGE_RE, RESULTS, ensemble_on
from ml.features import store

HARD_CAP_USD = 10.0
MAX_TOKENS = 500
# USD per million tokens (input, output)
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0)}
# cheap arms first so a cap hit cuts the expensive tail, not the baseline
ARMS = [("probes", None, None), ("haiku_blind", "claude-haiku-5-5", False), ("haiku_evidence", "claude-haiku-5-5", True),
        ("sonnet_blind", "claude-sonnet-5-5", False), ("sonnet_evidence", "claude-sonnet-5-5", True),
        ("opus_evidence", "claude-opus-5-5", True)]

SYSTEM = ("You decide whether an image is a fully AI-generated still. Judge visual evidence: anatomy, text, lighting, "
          "textures, physics, background coherence. Return the requested JSON only. explanation is at most 80 words. "
          "Image metadata was stripped, so absence of provenance means nothing.")


class BudgetExceeded(Exception):
    pass


def evidence_text(scores: dict, ens: float, band: str, th) -> str:
    per = ", ".join(f"{k}={v:.2f}" for k, v in scores.items())
    return (f"Detector evidence (probes trained to flag AI images; 1.0 = looks AI): {per}. Ensemble score {ens:.2f}, band '{band}'. "
            f"'likely_ai' means score >= {th.t_ai:.2f}, a cutoff that wrongly flags about 5% of real photos. "
            f"'likely_real' means score <= {th.t_real:.2f}, which still lets through about 10% of AI images. "
            "Between them is 'uncertain'. The detector can be wrong on new generators; set overrode_detector=true if you disagree with its band. "
            "Provenance: none available (metadata stripped).")


def stratified(rt: pd.DataFrame, n: int, seed=0) -> pd.DataFrame:
    """Proportional by (label, generator), at least 1 per cell; deterministic."""
    rt = rt.sample(frac=1, random_state=seed)
    cells = rt.groupby(["label", "generator"])
    want = {k: max(1, round(n * len(g) / len(rt))) for k, g in cells}
    return pd.concat([g.head(want[k]) for k, g in cells]).head(n).sort_values("image_id")


def est_cost(model: str, nbytes_pixels: int, text: str) -> float:
    pin, pout = PRICES[model]
    return ((nbytes_pixels / 750 + len(text) / 3 + 200) * pin + MAX_TOKENS * pout) / 1e6  # worst case: full max_tokens


class Cache:
    def __init__(self, path: Path):
        self.path, self.rows = path, {}
        if path.exists():
            for line in path.read_text().splitlines():
                r = json.loads(line)
                self.rows[(r["arm"], r["image_id"])] = r

    @property
    def spent(self) -> float:
        return sum(r["cost"] for r in self.rows.values())

    def add(self, r: dict):
        self.rows[(r["arm"], r["image_id"])] = r
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(r) + "\n")


def call_judge(backend, model, img_bytes, text) -> dict:
    try:
        out, (tin, tout) = backend.ask(model=model, system=SYSTEM, image=img_bytes, evidence=text,
                                       schema=anthropic.transform_schema(JudgeOutput.model_json_schema()),
                                       max_tokens=MAX_TOKENS, timeout=30)
    except BackendError as e:
        if e.reason == "retry_later":
            raise  # run_arms waits and retries the same image
        if e.reason == "refusal":
            return {"cost": 0.0, "verdict": "refusal"}
        return {"cost": 0.0, "verdict": "error", "error": e.reason}
    pin, pout = PRICES[model]
    return out.model_dump() | {"cost": (tin * pin + tout * pout) / 1e6}  # command backend reports (0, 0)


def run_arms(backend, slice_df, cache: Cache, cap: float, model_dir: Path, arms=ARMS) -> dict:
    meta = load_meta(model_dir / "meta.json")
    kept, ens, per = ensemble_on(model_dir, meta, slice_df, 0)
    info = {i: (float(ens[k]), {b: float(per[b][k]) for b in per}) for k, i in enumerate(kept.image_id)}
    stopped = None
    for arm, model, evidence in arms:
        if model is None:
            continue
        for r in slice_df.itertuples():
            if (arm, r.image_id) in cache.rows or (evidence and r.image_id not in info):
                continue
            img = store.resolve(r.path).read_bytes()
            text = "Is this image AI-generated?"
            if evidence:
                s, sc = info[r.image_id]
                text = evidence_text(sc, s, band_for(s, meta.thresholds), meta.thresholds) + "\n" + text
            if cache.spent + est_cost(model, len(img) * 4, text) > cap:
                stopped = f"cap ${cap:.2f} would be exceeded at arm {arm} (spent ${cache.spent:.4f})"
                return {"info": info, "meta": meta, "stopped": stopped}
            res = None
            while res is None:
                try:
                    res = call_judge(backend, model, img, text)
                except BackendError as e:  # only retry_later reaches here
                    wait = DEFAULT_RETRY_S if e.retry_after_s is None else e.retry_after_s
                    print(f"judge backend unavailable; waiting {wait:.0f}s then resuming")
                    time.sleep(wait)
                except Exception as e:  # noqa: BLE001  one bad image must not abort the run
                    res = {"cost": 0.0, "verdict": "error", "error": type(e).__name__}
            cache.add({"arm": arm, "image_id": r.image_id, "model": model, **res})
    return {"info": info, "meta": meta, "stopped": stopped}


def arm_table(slice_df, cache: Cache, ctx, arms=ARMS) -> list[dict]:
    rows = []
    y = dict(zip(slice_df.image_id, slice_df.label))
    for arm, model, evidence in arms:
        if model is None:
            v = {i: band_for(s, ctx["meta"].thresholds) for i, (s, _) in ctx["info"].items()}
            cost = 0.0
        else:
            v = {i: cache.rows[(arm, i)]["verdict"] for (a, i) in cache.rows if a == arm}
            cost = sum(r["cost"] for (a, _), r in cache.rows.items() if a == arm)
        if not v:
            continue
        f = [i for i in v if y[i] == 1]; rl = [i for i in v if y[i] == 0]
        ok = lambda i: v[i] == ("likely_ai" if y[i] else "likely_real")
        rows.append(dict(arm=arm, n=len(v), tpr=np.mean([v[i] == "likely_ai" for i in f]) if f else None,
                         fpr=np.mean([v[i] == "likely_ai" for i in rl]) if rl else None,
                         uncertain=np.mean([x == "uncertain" for x in v.values()]),
                         accuracy=np.mean([ok(i) for i in v]), failures=sum(x in ("error", "refusal") for x in v.values()), cost=cost))
    return rows


def write_block(rows, spent, cap, stopped, n, backend):
    f = lambda x: "n/a" if x is None else f"{x:.3f}"
    metered = "" if backend != "command" else " (cost is not metered for the command backend)"
    L = ["<!-- judge:start -->", "## Judge comparison", "",
         f"Stratified slice of report_test (n={n}), clean images. Verdict counted correct only when decisive and right (uncertain = wrong). "
         f"Spend ${spent:.4f} of ${cap:.2f} cap. Backend: {backend}{metered}." + (f" STOPPED: {stopped}." if stopped else ""), "",
         "| arm | n | TPR (ai called ai) | FPR (real called ai) | uncertain | accuracy | failures | cost $ |", "|---|---|---|---|---|---|---|---|"]
    L += [f"| {r['arm']} | {r['n']} | {f(r['tpr'])} | {f(r['fpr'])} | {f(r['uncertain'])} | {f(r['accuracy'])} | {r['failures']} | {r['cost']:.4f} |" for r in rows]
    block = "\n".join(L + ["<!-- judge:end -->"])
    txt = RESULTS.read_text() if RESULTS.exists() else "# Results\n\n"
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(JUDGE_RE.sub(lambda _: block, txt) if JUDGE_RE.search(txt) else txt + "\n" + block + "\n")


def main(argv=None, client=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--approve-spend", type=float, required=True, help="dollar cap you approve (max 10)")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--model-dir", default=None)
    a = ap.parse_args(argv)
    cap = min(a.approve_spend, HARD_CAP_USD)
    if cap <= 0:
        sys.exit("--approve-spend must be > 0")
    model_dir = (Path(a.model_dir) if a.model_dir else current_model_dir()).resolve()
    df = store.load_manifest()
    sl = stratified(df[df.split == "report_test"], a.n)
    cache = Cache(data_dir() / "eval" / "judge_cache.jsonl")
    if client is not None:
        backend = ApiBackend(client)
    else:
        kind = os.environ.get("SIXFINGERS_JUDGE_BACKEND") or "api"
        if kind == "api" and not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("ANTHROPIC_API_KEY is not set (or set SIXFINGERS_JUDGE_BACKEND=command)")
        # the API client reads ANTHROPIC_API_KEY from env; the key is never logged
        backend = from_env(anthropic.Anthropic() if kind == "api" else None)
    if backend is None:
        sys.exit("judge backend not configured (set SIXFINGERS_JUDGE_CMD for the command backend)")
    ctx = run_arms(backend, sl, cache, cap, model_dir)
    rows = arm_table(sl, cache, ctx)
    write_block(rows, cache.spent, cap, ctx["stopped"], len(sl), backend.name)
    print(f"spent ${cache.spent:.4f} / cap ${cap:.2f}" + (f"; stopped: {ctx['stopped']}" if ctx["stopped"] else ""))


if __name__ == "__main__":
    main()
