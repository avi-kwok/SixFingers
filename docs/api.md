# The API, explained for beginners

The API is a small web server (FastAPI). The website sends it a picture; it answers with a verdict. Code lives in `api/`.

## What happens to one picture (`POST /v1/check`)

0. **Input**. Send exactly one of the multipart file `image` or the form field `url` (an https link). Both or neither gives `400`. For a link, `api/fetch.py` downloads it first, in memory, then the bytes follow the same path as an upload (see **Fetching links**).
1. **Validate** (`api/validate.py`). We look at the first bytes of the file to see if it is really a JPEG, PNG or WebP (file names lie). More than 10 MB gives `413`. A wrong type gives `415`. A file with more than 25 million pixels is refused too, because tiny files can unpack into enormous images ("decompression bombs"). Nothing is written to disk.
2. **Provenance** (`api/provenance.py`). Some tools sign AI images with a C2PA manifest, or leave hints in metadata (like `trainedAlgorithmicMedia`). We report what we find. Finding nothing proves nothing, since metadata is easily stripped.
3. **Detector** (`api/detector.py`). Each backbone (a big pretrained vision model) turns the picture into a list of numbers (an embedding). A small "probe" per backbone turns that into a score from 0 to 1. We average the scores in logit space, giving one ensemble score. Two thresholds from `meta.json` turn it into a band: `likely_ai`, `uncertain` or `likely_real`.
4. **Judge** (`api/judge.py`). The judge starts at the same time as the detector, so total time is about the slower of the two. It sees the picture and the provenance findings, never the detector scores (in a 300-image comparison the image-only judge was more accurate). It returns a structured answer (verdict, confidence, cues, a short explanation). The backend is chosen by environment variable: by default Claude Haiku through the Anthropic API (low effort, 15 second timeout), or a local command (see below).
   **Evidence boxes.** Each cue may carry a `box`: `[x0, y0, x1, y1]`, where each number is a fraction (0 to 1) of the image's width or height, measured from the top-left corner. So `[0.5, 0, 1, 0.5]` is the top-right quarter. The judge sees the picture turned upright (EXIF rotation applied) and only shrunk, never cropped or stretched, so the same fractions mark the same spot on the upright original. The judge is asked for a tight box when a cue is a visible region and no box for whole-image cues (for example "overall lighting"). The server checks every box: numbers outside 0 to 1 are clamped to the edge, and a box that is malformed or empty (x0 must be below x1, y0 below y1) is dropped while its cue is kept. In the response, `judge.cues[].box` is `null` when there is no box. Clients written before boxes existed can ignore the field.
5. **Override flag**. `overrode_detector` is computed by the API: true when the judge gives a decisive verdict (`likely_ai` or `likely_real`) that differs from the detector band. The model's own value is ignored.
6. **Probability**. `probability_ai` is the number the site shows as the main figure. With `source: "judge"` it is Claude's confidence turned into P(AI) (`likely_ai` keeps its confidence, `likely_real` uses 1 minus it, `uncertain` is 0.5) and passed through the calibration map in `api/judge_calibration.json`. With `source: "detector_only"` it equals `detector.score`. See `docs/detector.md` section 9.
7. **Fallback**. If the judge is off, times out, errors, or the daily budget is spent, you still get the detector band with `source: "detector_only"` and a `judge_unavailable_reason`.
8. **Count and check_id**. The check adds 1 to today's usage total and gets a random `check_id` (a UUID) for feedback; see **Usage counts and feedback**. If that storage write fails, the error is logged, the check still succeeds, and `check_id` is `null`.

## Endpoints

| Endpoint | What it does |
|---|---|
| `POST /v1/check` | Multipart form with either `image` (a file) or `url` (an https link). Returns the JSON result, including `check_id`. |
| `POST /v1/feedback` | JSON feedback on one result: was the verdict right? See **Usage counts and feedback**. |
| `GET /v1/model` | Current model version, thresholds, and eval metrics. |
| `GET /v1/scorecard` | The current model's scorecard: per-generator catch rates and per-source false-alarm rates, plus live usage totals (see **Scorecard**). `404` if the model has none. |
| `GET /v1/health` | `200` with `status: "ok"`. `503` with `status: "loading"` while the backbones load, or `status: "no_model"` if no model is loadable. |

**Refusals.** If the judge refuses the image, the API returns `422` with detail `We can't analyse this image.` and no verdict. One `refusal` line is logged with the same fields as a check line.

Errors: `400` unreadable image, both or neither of `image` and `url`, or an unusable link, `413` too big, `415` wrong type, `422` the judge refused the image, `429` too many checks (see `Retry-After`), `502` the link could not be fetched, `503` no model, or busy (no free analysis slot within 5 seconds), `504` the link timed out.

Try it:

    make serve
    curl -F image=@photo.jpg http://127.0.0.1:8000/v1/check
    curl -F url=https://example.com/photo.jpg http://127.0.0.1:8000/v1/check

## Fetching links

The server downloads the link, so it must not be a way to reach internal machines. Rules in `api/fetch.py`:

- https only, no `user:pass@` in the URL.
- The host name is resolved and every address must be globally routable. Loopback, private, link-local (such as `169.254.169.254`), multicast, reserved, unspecified, carrier-grade NAT (`100.64.0.0/10`) and IPv4-mapped IPv6 forms of those are refused.
- The connection goes to the checked IP address, never to a second lookup, so DNS rebinding cannot swap the address. TLS server name, certificate check and the `Host` header still use the original host name.
- Up to 3 redirects, each followed by hand and checked with the same rules.
- 5 second connect timeout, 10 seconds in total. The body is streamed and stops at the 10 MB cap. No cookies; the `User-Agent` is `SixFingers/1.0`.
- The 10 seconds starts when the fetch begins and includes DNS lookups. Everything else (connect, TLS handshake, headers, every redirect hop and the body) is cut off at the deadline. A DNS lookup cannot be cut off, so a slow one can run past 10 seconds; the fetch then stops with `504` right after it. Socket timeouts alone restart on every byte, so a server sending one byte every few seconds could hold a worker forever. A timer shuts the connection down at the deadline instead, and the request gets `504`.
- Nothing is written to disk.

Errors: a blocked address, a non-https link or credentials give `400` "That link can't be used.". A `401`, `403`, `404` or `410` from the host gives `400` "Couldn't fetch that image. The link may have expired or not be public." (Discord links expire after about a day). A timeout gives `504`; any other fetch failure gives `502`. Link checks count toward the rate limit like uploads.

## Scorecard

`GET /v1/scorecard` returns `models/current/scorecard.json` as it is. The file is written offline, never by the server:

    .venv/bin/python -m ml.eval.scorecard --model-dir models/v16

It uses cached features only and does not change the model's other files. Contents:

- `model_version`, `date` (when the file was made) and `t_ai` (the threshold above which an image is flagged as AI).
- `detector.clean` and `detector.compressed`: the stored images, and the cached compressed copy of each.
  - `overall`: on `report_test`, `n`, `auroc`, `tpr_at_5fpr` (share of AI images caught when 5% of real photos are flagged) and `ece` (calibration error; lower is better).
  - `generators`: one row per AI generator in `report_test` (generators the probes never trained on), with `n` and `catch_rate`, the share of its images scored at or above `t_ai`.
  - `real_sources`: one row per real-photo source, with `n` and `false_alarm_rate`, the share wrongly scored at or above `t_ai`. RAISE comes from `report_test`. Rows marked `"split": "val"` (COCO, Open Images, Open Images originals) come from the validation split: the probes never trained on them, but `t_ai` was set so that at most 5% of val real photos score above it, so these rows sit near 5% by design. They show how the sources differ, not how unseen photos fare. Val has no compressed copies cached, so `compressed` lists RAISE only.
- `usage`: added by the server on every request, never stored in `scorecard.json`, so the numbers are always current. `checks_total` (all checks counted since launch), `checks_last_30_days` (today and the 29 days before, UTC), `feedback_answers` (how many feedback answers there are) and `feedback_right` (how many said "right"; `null` until there are at least 20 answers, so a handful of answers is never shown as a rate). `usage` is `null` if the usage file cannot be read.
- `judge`: the Claude judge on `report_test` (from `data/eval/judge_cache_prod.jsonl`, the answers for the current production prompt): `n`, `accuracy` (decisive and right; `uncertain` counts as wrong), `uncertain_rate`, and `calibrated_ece` from `api/judge_calibration.json`. `null` if those files are missing.

## Usage counts and feedback

Code in `api/usage.py`. Everything lives in one SQLite file, `usage.sqlite` in the data dir (`SIXFINGERS_DATA_DIR`, default `data/`), next to `judge_budget.json`, so restarts and code updates keep it. It has three tables:

- `daily_counts`: one row per (UTC date, input kind, verdict) with a running total `n`. Input kind is `file` or `link`. Every finished check calls `Usage.record_check(kind, verdict, probability, model_version, judge_used)` once, so other check routes (for example video or audio) only need that one call. A refused image (`422`) or an error is not counted.
- `checks`: for 24 hours only, `check_id` with the UTC date, input kind, verdict, the probability shown (`probability_ai`), the model version and whether the judge answered (`judge_used`, 1 or 0). This is what lets feedback be tied to a verdict without keeping the image. Rows older than 24 hours are deleted each time a check is counted, and are refused for feedback even before that.
- `feedback`: the answer, the optional `actual` label and comment, the `check_id` it answers (so it cannot be answered twice) and the verdict fields above.

Never stored: IP addresses or IP hashes, browser or device details, images or image hashes, links, or anything else per person. A storage error never fails a check: it is logged and the check returns `check_id: null`.

`POST /v1/feedback` takes a JSON body:

| Field | Rule |
|---|---|
| `check_id` | Required. The `check_id` from a check response (a version 4 UUID in lower case). |
| `answer` | Required. `right`, `wrong` or `unsure`. |
| `actual` | Optional. What the image really was: `ai`, `real` or `unknown`. |
| `comment` | Optional free text, at most 500 characters. Control characters are removed (line breaks are kept); a blank comment counts as none. |

Any other field, a wrong value or a longer comment gives `422`. Answers: `200` `{"status": "ok"}`, `404` the `check_id` is unknown or older than 24 hours, `409` this check already has feedback, `429` more than 20 feedback posts from one IP in an hour (same in-memory hashed limiter as checks; the IP is never stored), `503` the feedback could not be saved.

    curl -H 'Content-Type: application/json' -d '{"check_id": "<id>", "answer": "right"}' http://127.0.0.1:8000/v1/feedback

To read the data, run `make feedback` (or `.venv/bin/python -m ml.eval.feedback_report`). It prints the daily totals and writes every feedback row to `data/feedback.csv` (git-ignored, like all of `data/`). Comments that start with `=`, `+`, `-` or `@` get a leading `'` so a spreadsheet does not run them as formulas.

## Model loading and hot reload

`models/current` is a symlink to a folder such as `models/v3` holding one probe JSON per backbone plus `meta.json`. On every request the API checks where the link points. If it moved (the nightly loop promoted a new model, or you rolled back), the API loads the new folder with no restart. If the new folder is broken, it keeps serving the old model and logs the error.

The backbones themselves are loaded lazily on first use through `ml.features.backbones.load_backbone(name)`, whose result needs an `embed(list_of_PIL_images)` method. Tests replace this with a tiny stub, so they need no weights.

## Command backend contract

With `SIXFINGERS_JUDGE_BACKEND=command`, the API runs the program in `SIXFINGERS_JUDGE_CMD` once per judged image.

- **Request** (JSON on stdin): `model`, `system`, `evidence` (the provenance JSON as text, plus the question), `image` (`{"media_type": "image/jpeg", "data": <base64>}`), `schema` (the JSON schema of the verdict), `max_tokens`.
- **Verdict** (JSON on stdout, exit 0): the judge output fields `verdict`, `confidence`, `cues` (each with `region`, `cue` and an optional `box`), `explanation`, `overrode_detector`.
- **Temporarily unavailable**: exit 75 with stdout `{"retry_after_s": N}`. The API answers `detector_only` with reason `error`. `ml.judge_eval` waits `N` seconds (default 1800) and resumes from its cache.
- **Anything else** (other exit codes or invalid JSON) answers `detector_only` with reason `error`. No answer within 120 seconds answers `detector_only` with reason `timeout`.

Images are downscaled to at most 1568 px on the long edge, re-encoded as JPEG, and kept at 5 MB or less before any backend sees them.

## Limits and guardrails

- **Rate limit**: 10 checks per IP per hour. We store only a hash of the IP mixed with a secret salt and today's date, so it changes daily. It lives in memory, so a restart resets it (fine for one worker).
- **Judge budget**: about 500 Claude calls per day (`SIXFINGERS_JUDGE_BUDGET`). The counter is saved in `data/judge_budget.json`, so restarts do not reset it. If the detector fails, a judge call that has not started yet is cancelled and costs nothing; one that already ran stays counted, because it was paid for.
- **Judge share per IP**: one IP gets at most 20 judge calls per day (`SIXFINGERS_JUDGE_PER_IP_DAY`), so a few visitors cannot use up the whole daily budget. After that, the IP still gets checks, but as `source: "detector_only"` with `judge_unavailable_reason: "budget"`.
- **Image size**: at most 10 MB and 25 megapixels. A decoded image takes about 3 bytes per pixel in memory, so this cap keeps one check to a few hundred MB at most.
- **Concurrent checks**: at most 2 images are decoded and scored at the same time (`SIXFINGERS_MAX_CONCURRENT`). A check that cannot start within 5 seconds gets `503` "SixFingers is busy right now. Try again in a minute." The judge's downscaled copy (at most 1568 px) is made inside the slot; the judge call itself does not hold one.
- **Startup**: the backbones load in the background when the server starts. Until they are in memory, `GET /v1/health` answers `503` with `status: "loading"`, so a health check only passes once checks are fast.
- **CORS**: only `https://sixfingers.avikwok.dev` may call the API from a browser. Set `SIXFINGERS_DEV=1` to also allow `localhost:3000`.
- **Logs**: one JSON line per check (time, verdict, source, score, dimensions, format, latency, model version, IP hash, and `input`: `file` or `link`). For links only the host name is logged, never the full URL, which can carry a signed token. Never image bytes or image hashes. Every time the judge disagrees with the detector, an `override` line is logged; every refusal logs a `refusal` line.

## Settings (environment variables)

| Name | Meaning |
|---|---|
| `ANTHROPIC_API_KEY` | Enables the `api` judge backend. Unset means detector-only. Never put it in a file in the repo. |
| `SIXFINGERS_JUDGE_BACKEND` | Judge backend: `api` (default) or `command`. |
| `SIXFINGERS_JUDGE_CMD` | Command line for the `command` backend, for example `python3 /path/to/judge.py`. |
| `SIXFINGERS_JUDGE_BUDGET` | Daily judge calls, default 500. |
| `SIXFINGERS_JUDGE_PER_IP_DAY` | Judge calls per IP per day, default 20. |
| `SIXFINGERS_MAX_CONCURRENT` | Images decoded and scored at the same time, default 2. |
| `SIXFINGERS_IP_SALT` | Optional fixed salt; random per start if unset. |
| `SIXFINGERS_TRUST_PROXY` | Set to `1` behind Caddy so the real client IP is read from `X-Forwarded-For`. |
| `SIXFINGERS_CORS_ORIGINS` | Comma-separated allowed origins. |
| `SIXFINGERS_MODELS_DIR`, `SIXFINGERS_DATA_DIR` | Where models live, and where the budget file and `usage.sqlite` (usage counts and feedback) live. |

## Tests

`make test` runs `tests/api/`. A fake Anthropic client covers success, timeout, error, refusal (422) and budget cases, plus the image-only prompt, the computed override flag and detector/judge concurrency. `tests/api/test_boxes.py` covers box checks (clamping, dropping bad boxes, keeping the cue). `tests/api/test_scorecard.py` covers `GET /v1/scorecard` and builds a scorecard from a tiny fake manifest. A fake `models/vN` folder and a stub embedder cover the rest, including 413/415/429 and hot reload. `tests/api/test_fetch.py` covers links with mocked DNS and HTTP: blocked addresses, redirects, size cap, timeouts and the vetted-IP connection. `tests/api/test_usage.py` covers the usage counter (one count per check, no IP stored), feedback (success, `404`, `409`, `422`, the comment cap, expiry), the scorecard `usage` block below and above 20 answers, the CSV report, and a check that still succeeds when the usage file cannot be written. No network, no real API calls.
