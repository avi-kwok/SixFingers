# How the detector works (beginner guide)

The detector answers one question: does this image look like it came from an AI image generator? It never says "certainly". It gives a score between 0 and 1 and a band. Here are the pieces, in the order an image meets them.

## 1. Backbone: a pre-trained "eye"

A backbone is a large neural network that someone else already trained on hundreds of millions of images. We do not change it ("frozen"). Its job is to look at a picture and summarize it as numbers. We use three, because they learned to look at different things:

- `clip_vitl14` (OpenAI CLIP): learned from images paired with captions.
- `dinov2_vitl14` (Meta DINOv2): learned from images alone, good at texture and structure.
- `siglip` (Google SigLIP, base size): a CLIP-style model, smaller and faster.

Code: `ml/features/backbones.py`. All three expose `embed(images)`. On a Mac they run on the GPU (`mps`), on the server on the CPU.

## 2. Embedding: the picture as a list of numbers

The summary a backbone produces is called an embedding: a list of roughly 768 to 1024 numbers. Similar-looking images get similar lists. Nobody can read the numbers directly, but a simple model can learn which patterns mean "AI".

Computing embeddings is the slow part, so we cache them. `ml/features/store.py` saves one float16 file per (backbone, image, variant) under `data/features/`. Re-running extraction skips what exists. Variants: 0 is the clean image, 1 and 2 are augmented copies of training images (random crop, resize, blur, JPEG), 100 is the "compressed" copy of test images (resized, then JPEG quality 70) used to check the detector survives re-saving.

## 3. Probe: a tiny classifier on top

A probe is a small model trained on embeddings, with label 0 for real and 1 for AI.

- Logistic probe: one weight per number, added up, then squashed into 0..1. Training is quick and the result is a few KB of JSON.
- MLP probe: one hidden layer of 256 units. More flexible, so it can overfit. It is trained as a comparison and ships for a backbone only if it does better on the validation set.

Before training, each number is standardized (subtract the training average, divide by the spread). The saved probe stores those averages so the API does the same thing. Regularization strength (`C`, how strongly weights are pulled toward zero) is picked by trying a grid and keeping the value with the lowest validation log-loss. Code: `ml/probes/train.py`.

## 4. Calibration: making scores honest

A raw probe is often overconfident: it says 0.99 and is right only 80% of the time. Temperature scaling fixes this with a single number T: divide the raw score (the logit, the value before squashing) by T. We find the T that fits the validation set best. After this, "0.8" should mean roughly 80% of such images really are AI. We measure that with ECE (expected calibration error, 0 is perfect).

## 5. Ensemble: three opinions, one score

Each backbone gives a calibrated logit. We average the three and squash the average into 0..1. Averaging logits (not 0..1 scores) keeps one confident backbone from being drowned out by rounding.

## 6. Thresholds and bands

Two cutoffs turn the score into a band, both set on the validation set only:

- `t_ai`: chosen so that at most 5% of real validation images score at or above it. Score at or above `t_ai` is `likely_ai`. Pick a stricter cutoff and fewer real photos get falsely accused, but fewer AI images are caught.
- `t_real`: chosen so that at most 10% of AI validation images score at or below it. Score at or below `t_real` is `likely_real`.
- Anything in between is `uncertain`. This honest middle is deliberate.

## 7. How we measure it

`make eval` writes `docs/results.md` from cached features. The test set (`report_test`) contains a generator the training never saw, plus real camera photos (RAISE). Key numbers:

- TPR@5%FPR: of the AI images, what fraction is caught when only 5% of real images are wrongly flagged. The main number.
- AUROC: how well scores rank AI above real, ignoring any threshold (1.0 perfect, 0.5 coin flip).
- Clean vs compressed: the same test with images re-saved as JPEG, because shared images usually are.
- Per-generator TPR and real-image false-positive rate on RAISE.
- CPU latency per image, and a comparison against one open detector (`umm-maybe/AI-image-detector`, `ml/eval/baseline.py`).

The scorecard (`python -m ml.eval.scorecard`, served by `GET /v1/scorecard`) breaks these numbers down per generator and per real-photo source at `t_ai`, for clean and compressed images. See `docs/api.md`, section **Scorecard**.

## 8. The Claude judge comparison

`ml/judge_eval/run.py` tests whether asking a Claude model to look at the image beats the probes alone, and whether showing it the probe scores helps. Haiku, Sonnet and Opus are tried with and without the evidence on 300 test images. It costs real money, so it refuses to run without `--approve-spend 10`, stops before passing $10, and caches every answer. Tests use a fake client and never call the API.

The backend is chosen by the same switch the API uses: `SIXFINGERS_JUDGE_BACKEND=api` (default) or `command`, which runs the program in `SIXFINGERS_JUDGE_CMD`. Images are downscaled to at most 1568 px on the long edge, re-encoded as JPEG and kept under 5 MB before anything is sent. Exit code 75 from the command means "try later": the run waits its `retry_after_s` (default 1800 s) and resumes from its cache. The command backend's cost is not metered, so its rows show $0 spend.

## 9. Calibration

The detector score is already calibrated with temperature scaling. Claude's answer is calibrated separately. First, `python -m ml.judge_eval.run --split val --arms haiku_blind` asks Claude about 300 `val` images and caches the answers in `data/eval/judge_cache_prod.jsonl`, the default cache for both `run` and `calibrate`. That file always holds answers for the current production prompt. Answers from earlier prompts are kept for reference in `data/eval/judge_cache.jsonl` (the first eval prompt) and `data/eval/judge_cache_prod1.jsonl` (the production prompt before evidence boxes, which the current map was fitted on). The run sends exactly what the live API sends: the production prompt (`SYSTEM` in `api/judge.py`) and the provenance text read from the image itself, with no detector scores. A different prompt can change how confident the judge is, so a map fitted on one prompt may not carry over to another. After every prompt change we re-run the same images and refit only if the calibrated `report_test` ECE moves by more than 0.01 or accuracy drops by more than 1 point. Then `python -m ml.judge_eval.calibrate` converts each decisive answer to a raw P(AI) (`likely_ai` keeps its confidence, `likely_real` uses 1 minus it) and fits a monotone map to the true labels. It tries isotonic regression and Platt scaling, keeps the one with the lower cross-validated Brier score on `val` (Platt), and writes piecewise-linear knots to `api/judge_calibration.json` with the model, arm and n. Outputs are clipped to 2%-98%.

An `uncertain` answer is not fitted. The API maps it to 0.5, because Claude did not lean either way.

The map never sees `report_test`, so the `report_test` rows below are an honest out-of-sample check (294 decisive answers out of 300; the fit used 277 out of 300 on `val`). ECE uses 10 equal-width bins.

| | ECE | Brier |
|---|---|---|
| `val`, raw | 0.204 | 0.0694 |
| `val`, Platt (5-fold CV) | 0.018 | 0.0245 |
| `report_test`, raw | 0.229 | 0.0638 |
| `report_test`, calibrated | 0.028 | 0.0097 |

The table above is from the production prompt before evidence boxes (`judge_cache_prod1.jsonl`). Re-checked with the evidence-box prompt (same 300 `val` and 300 `report_test` images, now in `judge_cache_prod.jsonl`): the map still fits, so it was kept. On `report_test` the calibrated ECE is 0.025 (Brier 0.0128, raw ECE 0.218) and accuracy is 96.3% (was 96.7%).

When the judge model or prompt changes, move the old answers to a new file name, re-run both slices into `judge_cache_prod.jsonl`, and apply the rule above.

## Known limits

- Only fully AI-generated still images are in scope. Edited or upscaled images are not.
- The detector can only recognise styles it has seen; new generators can slip through. That is why the held-out generator is the headline test.
- Scores are evidence, not proof.
