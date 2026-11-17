# The API, explained for beginners

The API is a small web server (FastAPI). The website sends it a picture; it answers with a verdict. Code lives in `api/`.

## What happens to one picture (`POST /v1/check`)

0. **Input**. Send exactly one of the multipart file `image` or the form field `url` (an https link). Both or neither gives `400`. For a link, `api/fetch.py` downloads it first, in memory, then the bytes follow the same path as an upload (see **Fetching links**).
1. **Validate** (`api/validate.py`). We look at the first bytes of the file to see if it is really a JPEG, PNG or WebP (file names lie). More than 10 MB gives `413`. A wrong type gives `415`. A file with more than 40 million pixels is refused too, because tiny files can unpack into enormous images ("decompression bombs"). Nothing is written to disk.
2. **Provenance** (`api/provenance.py`). Some tools sign AI images with a C2PA manifest, or leave hints in metadata (like `trainedAlgorithmicMedia`). We report what we find. Finding nothing proves nothing, since metadata is easily stripped.
3. **Detector** (`api/detector.py`). Each backbone (a big pretrained vision model) turns the picture into a list of numbers (an embedding). A small "probe" per backbone turns that into a score from 0 to 1. We average the scores in logit space, giving one ensemble score. Two thresholds from `meta.json` turn it into a band: `likely_ai`, `uncertain` or `likely_real`.
4. **Judge** (`api/judge.py`). The judge starts at the same time as the detector, so total time is about the slower of the two. It sees the picture and the provenance findings, never the detector scores (in a 300-image comparison the image-only judge was more accurate). It returns a structured answer (verdict, confidence, cues, a short explanation). The backend is chosen by environment variable: by default Claude Haiku through the Anthropic API (low effort, 15 second timeout), or a local command (see below).
5. **Override flag**. `overrode_detector` is computed by the API: true when the judge gives a decisive verdict (`likely_ai` or `likely_real`) that differs from the detector band. The model's own value is ignored.
6. **Probability**. `probability_ai` is the number the site shows as the main figure. With `source: "judge"` it is Claude's confidence turned into P(AI) (`likely_ai` keeps its confidence, `likely_real` uses 1 minus it, `uncertain` is 0.5) and passed through the calibration map in `api/judge_calibration.json`. With `source: "detector_only"` it equals `detector.score`. See `docs/detector.md` section 9.
7. **Fallback**. If the judge is off, times out, errors, or the daily budget is spent, you still get the detector band with `source: "detector_only"` and a `judge_unavailable_reason`.

## Endpoints

| Endpoint | What it does |
|---|---|
| `POST /v1/check` | Multipart form with either `image` (a file) or `url` (an https link). Returns the JSON result. |
| `GET /v1/model` | Current model version, thresholds, and eval metrics. |
| `GET /v1/health` | `200 ok`, or `503` if no model is loadable. |

**Refusals.** If the judge refuses the image, the API returns `422` with detail `We can't analyse this image.` and no verdict. One `refusal` line is logged with the same fields as a check line.

Errors: `400` unreadable image, both or neither of `image` and `url`, or an unusable link, `413` too big, `415` wrong type, `422` the judge refused the image, `429` too many checks (see `Retry-After`), `502` the link could not be fetched, `503` no model, `504` the link timed out.

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
- Nothing is written to disk.

Errors: a blocked address, a non-https link or credentials give `400` "That link can't be used.". A `401`, `403`, `404` or `410` from the host gives `400` "Couldn't fetch that image. The link may have expired or not be public." (Discord links expire after about a day). A timeout gives `504`; any other fetch failure gives `502`. Link checks count toward the rate limit like uploads.

## Model loading and hot reload

`models/current` is a symlink to a folder such as `models/v3` holding one probe JSON per backbone plus `meta.json`. On every request the API checks where the link points. If it moved (the nightly loop promoted a new model, or you rolled back), the API loads the new folder with no restart. If the new folder is broken, it keeps serving the old model and logs the error.

The backbones themselves are loaded lazily on first use through `ml.features.backbones.load_backbone(name)`, whose result needs an `embed(list_of_PIL_images)` method. Tests replace this with a tiny stub, so they need no weights.

## Command backend contract

With `SIXFINGERS_JUDGE_BACKEND=command`, the API runs the program in `SIXFINGERS_JUDGE_CMD` once per judged image.

- **Request** (JSON on stdin): `model`, `system`, `evidence` (the provenance JSON as text, plus the question), `image` (`{"media_type": "image/jpeg", "data": <base64>}`), `schema` (the JSON schema of the verdict), `max_tokens`.
- **Verdict** (JSON on stdout, exit 0): the judge output fields `verdict`, `confidence`, `cues`, `explanation`, `overrode_detector`.
- **Temporarily unavailable**: exit 75 with stdout `{"retry_after_s": N}`. The API answers `detector_only` with reason `error`. `ml.judge_eval` waits `N` seconds (default 1800) and resumes from its cache.
- **Anything else** (other exit codes or invalid JSON) answers `detector_only` with reason `error`. No answer within 120 seconds answers `detector_only` with reason `timeout`.

Images are downscaled to at most 1568 px on the long edge, re-encoded as JPEG, and kept at 5 MB or less before any backend sees them.

## Limits and guardrails

- **Rate limit**: 10 checks per IP per hour. We store only a hash of the IP mixed with a secret salt and today's date, so it changes daily. It lives in memory, so a restart resets it (fine for one worker).
- **Judge budget**: about 500 Claude calls per day (`SIXFINGERS_JUDGE_BUDGET`). The counter is saved in `data/judge_budget.json`, so restarts do not reset it.
- **CORS**: only `https://sixfingers.avikwok.dev` may call the API from a browser. Set `SIXFINGERS_DEV=1` to also allow `localhost:3000`.
- **Logs**: one JSON line per check (time, verdict, source, score, dimensions, format, latency, model version, IP hash, and `input`: `file` or `link`). For links only the host name is logged, never the full URL, which can carry a signed token. Never image bytes or image hashes. Every time the judge disagrees with the detector, an `override` line is logged; every refusal logs a `refusal` line.

## Settings (environment variables)

| Name | Meaning |
|---|---|
| `ANTHROPIC_API_KEY` | Enables the `api` judge backend. Unset means detector-only. Never put it in a file in the repo. |
| `SIXFINGERS_JUDGE_BACKEND` | Judge backend: `api` (default) or `command`. |
| `SIXFINGERS_JUDGE_CMD` | Command line for the `command` backend, for example `python3 /path/to/judge.py`. |
| `SIXFINGERS_JUDGE_BUDGET` | Daily judge calls, default 500. |
| `SIXFINGERS_IP_SALT` | Optional fixed salt; random per start if unset. |
| `SIXFINGERS_TRUST_PROXY` | Set to `1` behind Caddy so the real client IP is read from `X-Forwarded-For`. |
| `SIXFINGERS_CORS_ORIGINS` | Comma-separated allowed origins. |
| `SIXFINGERS_MODELS_DIR`, `SIXFINGERS_DATA_DIR` | Where models and the budget file live. |

## Tests

`make test` runs `tests/api/`. A fake Anthropic client covers success, timeout, error, refusal (422) and budget cases, plus the image-only prompt, the computed override flag and detector/judge concurrency. A fake `models/vN` folder and a stub embedder cover the rest, including 413/415/429 and hot reload. `tests/api/test_fetch.py` covers links with mocked DNS and HTTP: blocked addresses, redirects, size cap, timeouts and the vetted-IP connection. No network, no real API calls.
