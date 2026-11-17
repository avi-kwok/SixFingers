"""Nightly loop: fetch -> extract -> retrain -> gate -> promote or reject -> report -> Discord.

Growth is capped per night: at most --grow new images per growable source from downloads, plus
--grow locally generated SDXL-Turbo images. Never reads report_test, never calls paid generation.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

from ml.common.paths import current_model_dir, data_dir, manifest_path, model_version_dir, models_dir
from ml.common.schemas import atomic_symlink
from ml.loop import gate, report


def versions() -> list[str]:
    return sorted((p.name for p in models_dir().glob("v*") if re.fullmatch(r"v\d+", p.name)), key=lambda v: int(v[1:]))


def live_version() -> str | None:
    c = current_model_dir()
    return Path(c.resolve()).name if c.is_symlink() and c.exists() else None


ALL = 10**6  # build every indexed image; growth is capped by the two steps before it


def sh(args: list[str]) -> None:
    if subprocess.run([sys.executable, "-m", *args]).returncode != 0:
        raise RuntimeError(f"{args[0]} failed")


def default_stages(grow: int) -> dict:
    return {
        "fetch": lambda: (sh(["ml.data.download", "--grow", str(grow)]),
                          sh(["ml.data.generate_local", "--grow", str(grow)]),
                          sh(["ml.data.build", "--skip-download", "--limit", str(ALL)])),
        "extract": lambda: sh(["ml.features.extract", "--limit", "0"]),
        "train": lambda: sh(["ml.probes.train"]),
    }


def count_rows() -> int:
    import pandas as pd

    return len(pd.read_parquet(manifest_path())) if manifest_path().exists() else 0


def run_loop(stages: dict, evaluate=None, rows=count_rows, out_dir: Path | None = None, post=report.post_discord) -> dict:
    """`stages` fetch/extract/train are callables; `evaluate(model_dir)` returns gate metrics."""
    r = {"date": date.today().isoformat(), "current_before": live_version(), "spend_usd": 0.0,
         "candidate": None, "gate": None, "outcome": "error"}
    r["current_after"] = r["current_before"]
    try:
        before_rows, before_v = rows(), set(versions())
        stages["fetch"]()
        stages["extract"]()
        r["images_added"] = rows() - before_rows
        stages["train"]()
        new = [v for v in versions() if v not in before_v]
        if not new:
            raise RuntimeError("train produced no new model version")
        cand = r["candidate"] = new[-1]
        cand_m = evaluate(model_version_dir(cand))
        cur_m = evaluate(model_version_dir(r["current_before"])) if r["current_before"] else None
        ok, reasons = gate.decide(cand_m, cur_m)
        r["gate"] = {"candidate_metrics": cand_m, "current_metrics": cur_m, "reasons": reasons}
        if ok:
            atomic_symlink(cand, current_model_dir())  # relative target keeps the symlink portable
            r["outcome"], r["current_after"] = "promoted", cand
        else:
            r["outcome"] = "rejected"
    except Exception as e:  # noqa: BLE001
        r["error"] = f"{type(e).__name__}: {e}"
    report.write_report(r, out_dir or data_dir() / "loop_reports")
    post(report.to_markdown(r))
    return r


def rollback(version: str | None = None) -> str:
    """Repoint models/current at `version`, or at the newest version older than the live one."""
    if version is None:
        live, vs = live_version(), versions()
        older = [v for v in vs if live and int(v[1:]) < int(live[1:])]
        if not older:
            raise SystemExit("no older version to roll back to")
        version = older[-1]
    if not model_version_dir(version).is_dir():
        raise SystemExit(f"{version} does not exist")
    atomic_symlink(version, current_model_dir())
    return version


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grow", type=int, default=200, help="max new images per source per night (downloads and local generation)")
    sub = ap.add_subparsers(dest="cmd")
    rb = sub.add_parser("rollback", help="repoint models/current at an older version")
    rb.add_argument("version", nargs="?")
    a = ap.parse_args()
    if a.cmd == "rollback":
        print("now live:", rollback(a.version))
        return 0
    import pandas as pd

    def evaluate(d: Path) -> dict:
        return gate.evaluate(d, pd.read_parquet(manifest_path()), gate.default_loader)

    r = run_loop(default_stages(a.grow), evaluate)
    print(report.to_markdown(r))
    return 0 if r["outcome"] != "error" else 1


if __name__ == "__main__":
    raise SystemExit(main())
