# SixFingers architecture (beginner guide)

SixFingers answers one question: "does this image look fully AI-generated?" It only covers whole images made by a generator, not photos that were edited or upscaled.

## The idea in plain words

A neural network ("backbone") turns an image into a list of numbers called an embedding. We never retrain the backbone. Instead a small classifier called a probe learns, from embeddings, which images are AI. A probe is either logistic regression (a weighted sum squashed into 0..1) or a small neural network with one hidden layer (an MLP). Several backbones each get a probe, and we average their opinions (the ensemble). A language model (Claude Haiku) then looks at the image and its metadata check, without the detector scores, and writes a short explanation. If it is unavailable, we show the detector's answer alone.

## The live model (v15)

`models/current` points at `models/v15`.

- Three backbones: `clip_vitl14`, `dinov2_vitl14`, `siglip`. Each has an MLP probe.
- Thresholds: `t_ai` = `t_real` = 0.227. `t_real` can never be above `t_ai`, so when the two rules overlap they meet at one value. Then the detector has no "uncertain" band: every score is either likely AI or likely real.
- Training data includes 130 camera photos from Open Images originals, so the probes learn what straight-from-camera photos look like.
- On report_test (clean): TPR@5%FPR 0.918, AUROC 0.981, 96% of AI images caught at `t_ai`, 18.1% of RAISE camera photos wrongly flagged.
- Compared with v13: fewer camera photos wrongly flagged (18.1% instead of 20.4%), the same share of AI images caught.

Full numbers: [results.md](results.md).

## Modules

| Path | Job |
|---|---|
| `ml/common/` | Shared contracts (paths, schemas, API models). Everyone imports these. |
| `ml/data/` | Source registry (`sources.yaml`), download, normalize, manifest, splits, shortcut audit. |
| `ml/features/` | Backbones (`embed(images) -> np.ndarray`) and the resumable feature cache. |
| `ml/probes/` | Train probes, calibrate, set thresholds, write `models/vN/`. |
| `ml/eval/` | Metrics and `docs/results.md`. |
| `ml/judge_eval/` | Budget-capped comparison of judge models. |
| `ml/loop/` | Nightly fetch, retrain, gate, promote. |
| `api/` | FastAPI service: validate, provenance, detector, judge. |
| `web/` | Next.js front end. |
| `deploy/` | VM setup, systemd, Caddy, runbook. |

## Data flow

1. `ml/data` downloads real and AI images, normalizes them (RGB, fixed size, same re-encoding for both sides so file format is not a giveaway), and writes `data/manifest.parquet`.
2. `ml/features` embeds each image with each backbone and caches float16 arrays in `data/features/`.
3. `ml/probes` trains one probe per backbone on `train`, calibrates on `val`, picks thresholds, and writes `models/vN/`.
4. `ml/eval` scores held-out `report_test` (one generator never seen in training) and writes `docs/results.md`.
5. `api` loads `models/current` (a symlink to `models/vN`), and reloads when the symlink target changes.

Splits: `train`, `val` (calibration and thresholds), `gate` (nightly promote/reject check), `report_test` (frozen, never used by the nightly loop).

## Contracts

All in `ml/common/`.

**paths.py**: `data_dir()`, `models_dir()` (env `SIXFINGERS_DATA_DIR`, `SIXFINGERS_MODELS_DIR`), plus `raw_dir()`, `normalized_dir()`, `features_dir()`, `manifest_path()`, `current_model_dir()`, `model_version_dir(v)`.

**schemas.py**:
- Manifest columns: `image_id` (sha256 of normalized bytes), `path`, `label` (0 real, 1 AI), `generator`, `source`, `split`, `license`.
- `ProbeArtifact` JSON: backbone, kind (`logistic` or `mlp`), embed_dim, mean, std, weights, bias, temperature, optional MLP weights. Score: standardize, linear, divide logit by temperature, sigmoid.
- `ModelMeta` (`models/vN/meta.json`): version, backbones, preprocess_version, thresholds `{t_ai, t_real}`, metrics, manifest hash, created_at.
- Helpers: `save_probe/load_probe`, `save_meta/load_meta`, `probe_logit`, `calibrated_logit`, `ensemble_score` (mean of calibrated logits, then sigmoid), `band_for(score, thresholds)`, `atomic_symlink(target, link)`.
- Model dir layout: `models/vN/{<backbone>.json, meta.json}`; `models/current -> vN`.

**api_models.py**: `CheckResponse` (the `POST /v1/check` body), `JudgeOutput` (what Claude returns), `DetectorResult`, `Provenance`, `Timings` (field `validate` is aliased; dump with `by_alias=True`).

## Thresholds

`t_ai` is set so only 5% of real validation images score at or above it. `t_real` is set so at most 10% of validation fakes score at or below it. When `t_real` is below `t_ai`, the scores between them are "uncertain" (v15 has no such band, because the two are equal).
