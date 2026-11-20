# Results (v16)

Written by `make eval` from cached features. report_test holds out one whole generator plus RAISE reals.

v16 is the live model (`models/current`). Its training data has 227 Open Images camera originals (167 train, 32 val, 28 gate), up from 130 in v15. Training used 25137 images, up from 20712.

Thresholds: t_ai=0.167, t_real=0.167

## Ensemble on report_test

| variant | n | TPR@5%FPR | AUROC | ECE | TPR@t_ai | real FPR@t_ai | RAISE FPR@t_ai |
|---|---|---|---|---|---|---|---|
| clean | 651 | 0.946 | 0.991 | 0.035 | 0.957 | 0.067 | 0.067 |
| compressed | 651 | 0.935 | 0.990 | 0.031 | 0.966 | 0.097 | 0.097 |

## Per-generator TPR at t_ai (clean)

- flux: 0.957

## Per backbone (clean)

| backbone | TPR@5%FPR | AUROC |
|---|---|---|
| clip_vitl14 | 0.946 | 0.987 |
| dinov2_vitl14 | 0.852 | 0.963 |
| siglip | 0.895 | 0.976 |

## Baseline (open HF detector, same report_test, clean)

| detector | n | TPR@5%FPR | AUROC |
|---|---|---|---|
| umm-maybe_AI-image-detector | 651 | 0.358 | 0.810 |

## Compared with v13 and v15

Same report_test, clean images unless noted.

| metric | v13 | v15 | v16 |
|---|---|---|---|
| t_ai | 0.251 | 0.227 | 0.167 |
| TPR@5%FPR | 0.915 | 0.918 | 0.946 |
| AUROC | 0.980 | 0.981 | 0.991 |
| ECE | 0.029 | 0.017 | 0.035 |
| TPR@t_ai | 0.960 | 0.960 | 0.957 |
| RAISE FPR@t_ai | 0.204 | 0.181 | 0.067 |
| RAISE FPR@t_ai (compressed) | 0.194 | 0.154 | 0.097 |
| dinov2_vitl14 TPR@5%FPR | 0.702 | 0.767 | 0.852 |
| siglip TPR@5%FPR | 0.852 | 0.881 | 0.895 |

What changed in v16:

- Far fewer RAISE camera photos are wrongly flagged as AI: 6.7% instead of 18.1% (clean), 9.7% instead of 15.4% (compressed).
- The share of AI images caught is about the same (0.957 at t_ai, 0.960 for v15). This is one generator (flux), so a 0.003 gap is within noise.
- Ranking is better: AUROC 0.991 instead of 0.981.
- Calibration is slightly worse: ECE 0.035 instead of 0.017.

Still a problem: 6.7% is above the 5% that `t_ai` aims for, and compressed photos are flagged more often (9.7%). Only 227 camera originals are in the training data so far.

## CPU latency per image (ms, batch 1)

Measured on v16 (2026-11-20, Apple M2 CPU, mean of 10 runs). v15 measured 893.3 ms total with the same backbones; the gap is run-to-run noise.

- clip_vitl14: 354.1
- dinov2_vitl14: 374.0
- siglip: 92.1
- ensemble_total: 820.2

<!-- judge:start -->
## Judge comparison

Run before v15 was trained. The `probes` arm and the `_evidence` arms used the scores of the model that was live then. Not rerun for v15 or v16.

The `haiku_blind` row uses the current production prompt (`SYSTEM` in `api/judge.py`, with evidence boxes) on the same 300 images. It was 0.967 accurate before evidence boxes were added, and 0.977 under the older eval prompt. The other judge arms still use the older eval prompt.

Stratified slice of report_test (n=300), clean images. Verdict counted correct only when decisive and right (uncertain = wrong). Spend $0.0000 of $10.00 cap. Backend: command (cost is not metered for the command backend).

| arm | n | TPR (ai called ai) | FPR (real called ai) | uncertain | accuracy | failures | cost $ |
|---|---|---|---|---|---|---|---|
| probes | 300 | 0.944 | 0.188 | 0.000 | 0.883 | 0 | 0.0000 |
| haiku_blind | 300 | 0.932 | 0.000 | 0.023 | 0.963 | 0 | 0.0000 |
| haiku_evidence | 300 | 0.944 | 0.000 | 0.017 | 0.963 | 0 | 0.0000 |
| sonnet_blind | 300 | 0.981 | 0.000 | 0.017 | 0.983 | 0 | 0.0000 |
| sonnet_evidence | 300 | 0.951 | 0.000 | 0.033 | 0.963 | 0 | 0.0000 |
| opus_evidence | 300 | 0.981 | 0.007 | 0.013 | 0.983 | 0 | 0.0000 |
<!-- judge:end -->
