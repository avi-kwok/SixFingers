# The nightly loop (beginner guide)

## What it is
Every night the server tries to make the detector a little better, and only keeps the result if it is provably not worse. Think of it as a nightly exam: a new student (the candidate model) sits the same exam as the current one, and replaces it only if it passes.

## Steps, in order
1. **Fetch**: adds new images in capped batches, then rebuilds the manifest.
   - `ml.data.download --grow N` adds up to N new images per growable source. Each source walks its pinned shard
     list in order and stops once N new images are added, so the work is bounded by N, not by shard size.
   - `ml.data.generate_local --grow N` makes up to N new SDXL-Turbo images from COCO captions. Paid image
     generation is never called here.
   - `ml.data.build --skip-download` rebuilds the manifest over every indexed image. Its shortcut audit runs too.
   - The held-out generator (`flux`) and RAISE never grow. Their report_test images are frozen (see [data.md](data.md)).
   - A failed download or a failed shortcut audit stops the night with an error report. The loop does not train
     on stale data.
2. **Extract**: `ml.features.extract --limit 0` turns every row of the manifest into embeddings (lists of numbers)
   and caches them. `--limit 0` means all rows; rows already cached are skipped.
3. **Retrain**: `ml.probes.train` fits fresh probes (small classifiers on the embeddings) and writes a new folder
   `models/vN+1`. Training never moves `models/current`.
4. **Gate**: the candidate and the current model are both scored on the **gate** split, a set of images used only for this decision. The candidate must pass all three checks:
   - TPR at 5% FPR is at least the current model's. (TPR: share of AI images caught. FPR: share of real images wrongly flagged. We fix FPR at 5% so the comparison is fair.)
   - No single generator's catch rate drops by more than 2 percentage points.
   - Real-image false positive rate stays at or below 5%.
5. **Promote or reject**:
   - Pass: `models/current` (a symlink, a shortcut pointing at a folder) is repointed to `vN+1` in one atomic step, so the API never sees a half-written state. Only this gate step promotes during the loop.
   - Fail: nothing changes. The rejected folder stays on disk for inspection.
   - A person can also promote by hand with `.venv/bin/python -m ml.probes.train --promote`. The make targets do that (`make train`, `make smoke`, `make full`).
6. **Report**: `data/loop_reports/YYYY-MM-DD.json` and `.md` (images added, metrics, gate result, spend), and a short message to Discord.

The loop never reads `report_test`, the frozen split used for the published numbers. Using it for decisions would quietly make those numbers optimistic.

## When it runs
Cron starts it at 02:00 Pacific. `flock` stops two copies running at once, `timeout 230m` kills it before 06:00, `nice` keeps the API responsive. Config lives in `/etc/cron.d/sixfingers-loop` (written by `deploy/setup.sh`). Cron runs `deploy/run_loop.sh`, which loads `.env` and passes `--grow` from `LOOP_GROW` in `.env` (default 200). `LOOP_GROW` sets the cap for both the download and the local generation.

## Running and undoing
```bash
.venv/bin/python -m ml.loop.run --grow 50         # run once by hand, capped at 50 new images per source
.venv/bin/python -m ml.loop.run rollback          # go back one version
.venv/bin/python -m ml.loop.run rollback v2       # go to a specific version
```
Rollback only repoints the symlink; the API reloads on its own.

## Discord
Set `DISCORD_WEBHOOK_URL` in `.env`. The URL is a secret: the code never prints it, and any error message is scrubbed of it. No URL means no post; the report files are still written.

## Code map
| File | Job |
|---|---|
| `ml/loop/run.py` | The pipeline and the rollback command |
| `ml/loop/gate.py` | Scoring and the pass/fail rules |
| `ml/loop/report.py` | JSON, Markdown, Discord |
| `deploy/run_loop.sh` | What cron actually runs (loads `.env`) |
| `tests/test_loop.py` | Gate, promote, reject, rollback and error paths using fake models |
| `tests/test_ml_pipeline.py::test_train_never_moves_current_only_gate_promotes` | Plain training never moves `models/current`; only the gate's promote does |

## Integration points
The gate needs the cached embeddings for the gate images. `default_loader` in `ml/loop/gate.py` reads one `.npy` file per image through `ml.features.store.feature_path(backbone, image_id, 0)`. The trainer writes a new `models/vN` directory and never moves `models/current` itself; the promote step is the only place that does.
