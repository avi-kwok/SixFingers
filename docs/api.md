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
| `POST /v1/check-video` | Multipart form with `video` (MP4, WebM or MOV, up to 100 MB and 60 seconds). Returns the video result, including `check_id`. See below and [video.md](video.md). |
| `POST /v1/check-audio` | Prototype, only when `SIXFINGERS_ENABLE_AUDIO=1` (otherwise `404`). Multipart form with `audio` (WAV, MP3, FLAC, OGG, M4A or WebM, up to 25 MB and 60 seconds). Returns `check_id`. See **Audio checks** below. |
| `POST /v1/feedback` | JSON feedback on one result (image, video or audio): was the verdict right? See **Usage counts and feedback**. |
| `GET /v1/model` | Current model version, thresholds, and eval metrics. |
| `GET /v1/scorecard` | The current model's scorecard: per-generator catch rates and per-source false-alarm rates, plus live usage totals (see **Scorecard**). `404` if the model has none. |
| `GET /v1/health` | `200` with `status: "ok"`. `503` with `status: "loading"` while the backbones load, or `status: "no_model"` if no model is loadable. With audio checks on, it also reports `audio_enabled` (`true` once the audio model has loaded); the website shows its Audio tab only then. |

**Refusals.** If the judge refuses the image, the API returns `422` with detail `We can't analyse this image.` and no verdict. One `refusal` line is logged with the same fields as a check line.

Errors: `400` unreadable image, both or neither of `image` and `url`, or an unusable link, `413` too big, `415` wrong type, `422` the judge refused the image, `429` too many checks (see `Retry-After`), `502` the link could not be fetched, `503` no model, or busy (no free analysis slot within 5 seconds), `504` the link timed out.

Try it:

    make serve
    curl -F image=@photo.jpg http://127.0.0.1:8000/v1/check
    curl -F url=https://example.com/photo.jpg http://127.0.0.1:8000/v1/check

## What happens to one video (`POST /v1/check-video`)

A video is checked from up to 20 still frames, not from the whole file. The full walk-through, with the reasons, is in [video.md](video.md). In short: the upload is validated and probed with `ffprobe`, `ffmpeg` pulls 16 frames at the centres of 16 equal slices plus up to 4 at scene cuts, the detector scores each frame, and the video score is the mean of the 4 highest frame scores. Claude reads the labelled frame grids at the same time. Audio is not analysed unless audio checks are on (see **Audio checks** below).

The response:

| Field | Meaning |
|---|---|
| `verdict`, `source` | The picture verdict, as for images: `likely_ai`, `uncertain` or `likely_real`; `source` is `judge` or `detector_only`. It comes from the frames only. |
| `overall_verdict`, `overall_flagged_by` | The picture verdict and the voice result combined (rule below). `overall_flagged_by` lists which parts said `likely_ai`: `["picture"]`, `["voice"]`, both, or empty. |
| `check_id` | Random id for `POST /v1/feedback`, or `null` if the usage file could not be written. |
| `judge` | `null` when the judge did not answer. Otherwise `model`, `confidence`, `explanation`, `overrode_detector`, and `cues`: a list of `{frame, time_s, region, cue}`. `frame` is 1 to 20, and `time_s` is looked up by the server from that frame. |
| `judge_unavailable_reason` | `budget`, `timeout`, `error`, `refusal`, `disabled` or `verification` when the judge did not answer; otherwise `null`. |
| `detector` | `score` (the mean of the top 4 frame scores), `band`, `thresholds` (`t_ai`, `t_real`), and `frames`: a list of `{index, time_s, score, band}` in time order. `index` is 1 to 20, as labelled in the grid. |
| `audio` | With audio checks off: `{analysed: false, note: "Audio is not analysed yet."}`. With them on: the audio score of the soundtrack (same fields as `audio` in **Audio checks** below), or `analysed: false` with a note saying the video has no audio track or that the track could not be read. |
| `duration_s`, `model_version` | Video length in seconds and the detector model version. |
| `timings_ms` | `validate`, `frames`, `detector`, `judge` and `total`, in milliseconds. |
| `disclaimer` | The estimate-not-proof note for videos. |

**The overall verdict.** It is a plain rule, not a trained model:

- If the picture says `likely_ai`, or the voice says `likely_ai`, the overall verdict is `likely_ai`, and `overall_flagged_by` says which part did it.
- If the picture says `likely_real` and the voice (when there is one) says `likely_real`, the overall verdict is `likely_real`.
- Anything else is `uncertain`.
- With no voice result (audio checks off, no soundtrack, or an unreadable one), the overall verdict is the picture verdict.

Flagging on either part raises false alarms. On 150 real test videos the judge wrongly flagged 1.3%, and the detector, used when the judge is unavailable, 5.3%. The voice check wrongly flagged 3.3% of real speech on In-the-Wild. Put together, up to about 9% of real videos can be flagged when only the detector is available. Accuracy details: [video.md](video.md). The voice check is a prototype.

Errors: `400` missing `video` field, empty upload, unreadable video, or WebM without a recorded length. `413` over 100 MB, over 60 seconds, or frames over 40 megapixels. `415` not MP4, WebM or MOV. `422` the judge refused the video, with detail `We can't analyse this video.` and no verdict. `429` too many checks (see `Retry-After`). `503` when `ffmpeg` or `ffprobe` cannot be found, or no model is loadable.

Limits: a video counts as 3 checks against the rate limit, and uses one judge call. Frames are judged, not motion or audio.

Try it:

    curl -F video=@clip.mp4 http://127.0.0.1:8000/v1/check-video

## Audio checks (prototype, `POST /v1/check-audio`)

Audio checks are off by default, so the public site behaves exactly as without them. Turn them on with `SIXFINGERS_ENABLE_AUDIO=1`. They answer one question: does this speech sound machine-made? How the model was built and how well it does is in [audio.md](audio.md).

What happens to one audio file (`api/audio.py`):

1. **Validate**. The upload is saved to a temporary folder that is deleted afterwards. Its first bytes must look like WAV, MP3, FLAC, OGG, M4A or WebM (file names lie). Over 25 MB gives `413`, an empty upload `400`, anything else `415`.
2. **Decode**. `ffmpeg` turns the file into 16 kHz mono samples, with the same safety rules as video: the format is forced from the sniffed bytes, only local files may be opened, and there is a 30 second time limit. Longer than 60 seconds gives `413`, an unreadable file `400`.
3. **Score**. Silence at the start and end is trimmed and the first 4 seconds are kept, exactly as in training. The speech model (XLS-R, about 1.2 GB) turns them into an embedding, and a small probe turns that into `p_fake`, the chance that the voice is AI-generated. The speech model loads on the first audio check, so that one is slower.
4. **Band**. `p_fake` at or above the model's threshold (from its `meta.json`) is `likely_ai`; below it, `likely_real`. There is no `uncertain` band yet.

There is no judge step: Claude cannot hear audio, so `source` is always `detector_only` and the note says so.

The response:

| Field | Meaning |
|---|---|
| `verdict` | The audio band: `likely_ai` or `likely_real`. |
| `source` | Always `detector_only`. |
| `audio` | `analysed: true`, `p_fake` (0 to 1), `band`, `threshold`, `model_version` (for example `a3`), and `note` (the prototype limits). |
| `duration_s` | Length of the decoded audio in seconds. |
| `check_id` | Random id for `POST /v1/feedback`, or `null`. |
| `timings_ms` | `validate`, `detector` and `total`, in milliseconds. |
| `disclaimer` | The estimate-not-proof note for audio. |

Errors: `400` missing `audio` field, empty upload or unreadable audio. `413` over 25 MB or over 60 seconds. `415` not one of the supported formats. `429` too many checks (an audio file counts as one). `503` when the audio model or `ffmpeg` is not available.

Videos: with audio checks on, `POST /v1/check-video` also decodes the soundtrack (while the frames are being sampled) and scores it with the same model (while the frames are being scored). `verdict` still comes from the frames only; the audio score is shown next to it, and `overall_verdict` combines the two (see above).

Try it:

    SIXFINGERS_ENABLE_AUDIO=1 make serve
    curl -F audio=@voice.wav http://127.0.0.1:8000/v1/check-audio

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

- `daily_counts`: one row per (UTC date, input kind, verdict) with a running total `n`. Input kind is `file`, `link`, `video` or `audio`. Every finished check calls `Usage.record_check(kind, verdict, probability, model_version, judge_used)` once. A video is counted with its overall verdict and the picture score; an audio check with its band and `p_fake`. The scorecard usage totals count every kind. A refused image (`422`) or an error is not counted.
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

A video request sends the same six keys, once per video. `image` is the composed frame-grid JPEG (up to three grids of up to 8 labelled frames each, the third for scene-cut frames, at most 1568 px on the long edge), and `schema` is the video verdict schema, whose cues carry `frame` as well as `region` and `cue`.

## Limits and guardrails

- **Rate limit**: 10 checks per IP per hour, where a video counts as 3. We store only a hash of the IP mixed with a secret salt and today's date, so it changes daily. It lives in memory, so a restart resets it (fine for one worker).
- **Judge budget**: about 500 Claude calls per day (`SIXFINGERS_JUDGE_BUDGET`). The counter is saved in `data/judge_budget.json`, so restarts do not reset it. If the detector fails, a judge call that has not started yet is cancelled and costs nothing; one that already ran stays counted, because it was paid for.
- **Judge share per IP**: one IP gets at most 20 judge calls per day (`SIXFINGERS_JUDGE_PER_IP_DAY`), so a few visitors cannot use up the whole daily budget. After that, the IP still gets checks, but as `source: "detector_only"` with `judge_unavailable_reason: "budget"`.
- **Image size**: at most 10 MB and 25 megapixels. A decoded image takes about 3 bytes per pixel in memory, so this cap keeps one check to a few hundred MB at most.
- **Concurrent checks**: at most 2 images are decoded and scored at the same time (`SIXFINGERS_MAX_CONCURRENT`). A check that cannot start within 5 seconds gets `503` "SixFingers is busy right now. Try again in a minute." The judge's downscaled copy (at most 1568 px) is made inside the slot; the judge call itself does not hold one.
- **Startup**: the backbones load in the background when the server starts. Until they are in memory, `GET /v1/health` answers `503` with `status: "loading"`, so a health check only passes once checks are fast.
- **CORS**: only `https://sixfingers.avikwok.dev` may call the API from a browser. Set `SIXFINGERS_DEV=1` to also allow `localhost:3000`.
- **Logs**: one JSON line per check (time, verdict, source, score, dimensions, format, latency, model version, IP hash, and `input`: `file`, `link`, `video` or `audio`; video lines also carry `duration_s` and `frames`, audio lines `duration_s`). For links only the host name is logged, never the full URL, which can carry a signed token. Never image bytes or image hashes, and never video bytes or frames. Every time the judge disagrees with the detector, an `override` line is logged; every refusal logs a `refusal` line.

## Settings (environment variables)

| Name | Meaning |
|---|---|
| `ANTHROPIC_API_KEY` | Enables the `api` judge backend. Unset means detector-only. Never put it in a file in the repo. |
| `SIXFINGERS_JUDGE_BACKEND` | Judge backend: `api` (default) or `command`. |
| `SIXFINGERS_JUDGE_CMD` | Command line for the `command` backend, for example `python3 /path/to/judge.py`. |
| `SIXFINGERS_JUDGE_BUDGET` | Daily judge calls, default 500. |
| `SIXFINGERS_JUDGE_PER_IP_DAY` | Judge calls per IP per day, default 20. |
| `SIXFINGERS_TURNSTILE_SECRET` | Cloudflare Turnstile secret key. Unset (default) means no bot check. When set, image, link and video checks need a token in the `cf-turnstile-response` request header; without a valid one the answer is detector-only with `judge_unavailable_reason: "verification"` (never an error, and no judge budget or IP quota is used). Audio has no judge, so it is not checked. The visitor's IP is not sent to Cloudflare. |
| `SIXFINGERS_TURNSTILE_HOSTNAMES` | Comma-separated hostnames Cloudflare may report for a Turnstile token (a subdomain of an entry also matches). Default: the production web host, `sixfingers.avikwok.dev`. A valid token for another hostname counts as failed verification. With `SIXFINGERS_DEV=1`, `localhost` and `example.com` (the host Cloudflare's test keys return) are also allowed. |
| `SIXFINGERS_MAX_CONCURRENT` | Images decoded and scored at the same time, default 2. |
| `SIXFINGERS_IP_SALT` | Optional fixed salt; random per start if unset. |
| `SIXFINGERS_TRUST_PROXY` | Set to `1` behind Caddy so the real client IP is read from `X-Forwarded-For`. |
| `SIXFINGERS_CORS_ORIGINS` | Comma-separated allowed origins. |
| `SIXFINGERS_MODELS_DIR`, `SIXFINGERS_DATA_DIR` | Where models live, and where the budget file and `usage.sqlite` (usage counts and feedback) live. |
| `SIXFINGERS_FFMPEG` | Path to the `ffmpeg` binary used for video checks. Default: found on `PATH`. |
| `SIXFINGERS_FFPROBE` | Path to the `ffprobe` binary used for video checks. Default: found on `PATH`. |
| `SIXFINGERS_ENABLE_AUDIO` | Set to `1` to turn on the prototype audio checks (`POST /v1/check-audio` and the audio score in video results). Default: off. |
| `SIXFINGERS_AUDIO_MODEL` | Which audio model folder under `models/audio/` to use. Default: `a3`. |

## Tests

`make test` runs `tests/api/`. `tests/api/test_turnstile.py` covers the bot check with a mocked Cloudflare answer (only Cloudflare's published test secret keys appear). A fake Anthropic client covers success, timeout, error, refusal (422) and budget cases, plus the image-only prompt, the computed override flag and detector/judge concurrency. `tests/api/test_boxes.py` covers box checks (clamping, dropping bad boxes, keeping the cue). `tests/api/test_scorecard.py` covers `GET /v1/scorecard` and builds a scorecard from a tiny fake manifest. A fake `models/vN` folder and a stub embedder cover the rest, including 413/415/429 and hot reload. `tests/api/test_fetch.py` covers links with mocked DNS and HTTP: blocked addresses, redirects, size cap, timeouts and the vetted-IP connection. `tests/api/test_usage_av.py` covers the counter and feedback for video and audio checks and the overall-verdict rule. `tests/api/test_usage.py` covers the usage counter (one count per check, no IP stored), feedback (success, `404`, `409`, `422`, the comment cap, expiry), the scorecard `usage` block below and above 20 answers, the CSV report, and a check that still succeeds when the usage file cannot be written. No network, no real API calls.

Video tests: `tests/api/test_video_ffmpeg.py` and `tests/api/test_video_api.py`. Clips are made with ffmpeg `testsrc` during the test, the judge and the detector are mocked, and the tests are skipped when ffmpeg is not installed. The playlist test checks that an HLS or concat file pointing at `http://` is refused without any connection.

Audio tests: `tests/api/test_audio_api.py`. Tones are made with ffmpeg in every supported format, and a fake embedder stands in for XLS-R, so no test needs the GPU. They check that the route does not exist with the flag off, the caps (`400`, `413`, `415`), videos with and without a soundtrack, and that the speech model loads only once.
