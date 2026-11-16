"""Build the manifest: download (optional) -> normalize -> split -> audit.
report_test is frozen once by an explicit step and never changes afterwards; later builds (any --limit)
keep exactly the frozen rows and only append to train/val/gate.
    python -m ml.data.build --limit 50                 # smoke scale
    python -m ml.data.build --limit N --freeze         # one-time freeze of report_test (full scale)
Generated fakes (SDXL-Turbo) come from `python -m ml.data.generate_local`, run separately."""
import argparse
import sys

import pandas as pd

from ml.common.paths import manifest_path
from ml.common.schemas import MANIFEST_COLUMNS
from ml.data import audit as audit_mod
from ml.data.download import DOWNLOADERS
from ml.data.normalize import Cache
from ml.data.sources import enabled_sources, holdout_generator, load_registry, read_index, source_dir
from ml.data.splits import assign_splits, freeze, frozen_path, read_frozen


def build_manifest(limit: int, skip_download: bool = False) -> pd.DataFrame:
    reg = load_registry()
    cache, rows, seen = Cache(), [], set()
    for s in enabled_sources(reg):
        fn = DOWNLOADERS.get(s["downloader"])
        if fn and not skip_download:
            print(f"[{s['name']}] download up to {limit}", flush=True)
            fn(s, limit)
        for r in read_index(s["name"])[: min(limit, s.get("max_limit", limit))]:
            raw = source_dir(s["name"]) / r["file"]
            key = f"{s['name']}/{r['file']}"
            hit = cache.get(key)
            image_id, path = (cache.map[key], hit) if hit else cache.put(key, raw)
            if image_id in seen:  # same pixels twice (e.g. a reshared image): keep one
                continue
            seen.add(image_id)
            rows.append(dict(image_id=image_id, path=str(path), label=int(s["side"] == "ai"),
                             generator=s["generator"], source=s["name"], license=r.get("license_url") or s["license"],
                             group=r["group"]))
    cache.save()
    df = pd.DataFrame(rows)
    df["split"] = assign_splits(df, holdout_generator(reg))
    frozen = read_frozen()
    # report_test never changes after the freeze step; new held-out/RAISE images are dropped, never trained on
    if frozen is not None:
        df = df[(df.split != "report_test") | df.image_id.isin(frozen)]
    # prompts are shared across generators: drop train/val/gate rows whose content also sits in report_test
    leak = (df.split != "report_test") & df.group.isin(df.loc[df.split == "report_test", "group"])
    return df[~leak].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=200, help="images per source")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--freeze", action="store_true", help="write the frozen report_test list (once, ever)")
    a = ap.parse_args()
    df = build_manifest(a.limit, a.skip_download)
    if a.freeze:
        _, wrote = freeze(df.loc[df.split == "report_test", "image_id"].tolist())
        if wrote:
            print(f"froze {(df.split == 'report_test').sum()} report_test ids -> {frozen_path()}")
        else:
            print(f"report_test already frozen ({frozen_path()}); list unchanged")
    frozen = read_frozen()
    print(f"report_test: {(df.split == 'report_test').sum()} rows")
    if frozen is not None:
        print(f"frozen list: {len(frozen)} ids")
    else:
        print("report_test is not frozen yet; freeze once at full scale with: python -m ml.data.build --limit N --freeze")
    manifest_path().parent.mkdir(parents=True, exist_ok=True)
    df[MANIFEST_COLUMNS].to_parquet(manifest_path(), index=False)
    print(pd.crosstab([df.label, df.generator], df.split).to_string())
    print(f"wrote {manifest_path()} ({len(df)} rows)")
    res = audit_mod.audit(df)
    print(f"shortcut audit AUROC={res['auroc']:.3f} (limit {audit_mod.MAX_AUROC}): {res['single_feature_auroc']}; "
          f"file_size AUROC={res['file_size_auroc']:.3f} (not gated)")
    if not res["passed"]:
        sys.exit("FAIL: shortcut audit; trivial features predict the label")


if __name__ == "__main__":
    main()
