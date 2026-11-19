# Results (v15)

Written by `make eval` from cached features. report_test holds out one whole generator plus RAISE reals.

v15 is the live model (`models/current`). Its training data adds 130 Open Images camera originals (94 train, 19 val, 17 gate) to the earlier sources.

Thresholds: t_ai=0.227, t_real=0.227

## Ensemble on report_test

| variant | n | TPR@5%FPR | AUROC | ECE | TPR@t_ai | real FPR@t_ai | RAISE FPR@t_ai |
|---|---|---|---|---|---|---|---|
| clean | 651 | 0.918 | 0.981 | 0.017 | 0.960 | 0.181 | 0.181 |
| compressed | 651 | 0.918 | 0.979 | 0.027 | 0.963 | 0.154 | 0.154 |

## Per-generator TPR at t_ai (clean)

- flux: 0.960

## Per backbone (clean)

| backbone | TPR@5%FPR | AUROC |
|---|---|---|
| clip_vitl14 | 0.920 | 0.977 |
| dinov2_vitl14 | 0.767 | 0.931 |
| siglip | 0.881 | 0.962 |

## Baseline (open HF detector, same report_test, clean)

| detector | n | TPR@5%FPR | AUROC |
|---|---|---|---|
| umm-maybe_AI-image-detector | 651 | 0.358 | 0.810 |

## Compared with v13

v13 was trained without camera originals. Same report_test, clean images unless noted.

| metric | v13 | v15 |
|---|---|---|
| t_ai | 0.251 | 0.227 |
| TPR@5%FPR | 0.915 | 0.918 |
| AUROC | 0.980 | 0.981 |
| ECE | 0.029 | 0.017 |
| TPR@t_ai | 0.960 | 0.960 |
| RAISE FPR@t_ai | 0.204 | 0.181 |
| RAISE FPR@t_ai (compressed) | 0.194 | 0.154 |
| dinov2_vitl14 TPR@5%FPR | 0.702 | 0.767 |
| siglip TPR@5%FPR | 0.852 | 0.881 |

What changed:

- Fewer RAISE camera photos are wrongly flagged as AI: 18.1% instead of 20.4% (clean), 15.4% instead of 19.4% (compressed).
- The share of AI images caught stays the same (0.960 at t_ai).
- Scores are better calibrated (ECE 0.017 instead of 0.029).

Still a problem: `t_ai` aims for 5% false alarms on val, but about 18% of RAISE camera photos still score above it. Only 130 camera originals are in the training data so far.

## CPU latency per image (ms, batch 1)

Measured on v13. v15 uses the same three backbones, so this time is the same. Not re-measured for v15.

- clip_vitl14: 351.1
- dinov2_vitl14: 400.9
- siglip: 91.7
- ensemble_total: 843.7

<!-- judge:start -->
## Judge comparison

Run before v15 was trained. The `probes` arm and the `_evidence` arms used the scores of the model that was live then. Not rerun for v15.

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
