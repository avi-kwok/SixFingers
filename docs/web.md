# Web front end (beginner guide)

The site in `web/` is a thin client. It never analyses images itself. It sends the file to `POST /v1/check` and draws the JSON that comes back.

Flow: pick, drop or paste an image, or paste an https image link into the "or paste an image link" field next to the upload control and press Check (pasting plain text that is an https URL anywhere on the page fills that field). The link is sent as the form field `url` and the server fetches it. Then the browser checks type and size (10 MB), shows a preview and uploads with a progress bar. The result card shows the verdict band, then "X% likely AI" (`probability_ai`) as the main number with "Detector score: Y%" smaller beneath it, then the explanation and cues, a "Detector only" tag when the explainer was unavailable, a note when it overrode the detector (for example "The detector leaned AI, but Claude's visual check found it real."), a C2PA badge, the scope note and the disclaimer.

Errors map to HTTP status: 400 for a link ("That link can't be used.", or "Couldn't fetch that image. The link may have expired or not be public."), 504 the link timed out, 502 the link could not be fetched, 429 rate limit, 413 too big, 415 wrong type, 422 the image cannot be analysed ("We can't analyse this image."). A line under the upload control says uploading means agreeing to the Terms. `/terms` is a short acceptable-use page, linked in the footer next to Privacy. A spent daily budget arrives as `judge_unavailable_reason: "budget"` on a normal 200 response.

The upload goes direct to the API because Vercel limits request bodies to about 4.5 MB. Set `NEXT_PUBLIC_API_URL` for the target API, or `NEXT_PUBLIC_MOCK=1` to try the UI with no server. Details: `web/README.md`.

Keep `web/lib/types.ts` in step with `ml/common/api_models.py`.

The About page explains the two percentages in plain words.

## Result card extras

- **Provenance first.** When the image has a C2PA manifest, a banner ("Signed by <issuer>: marked AI-generated", or "Signed by <issuer>") sits above the verdict. Without one, provenance stays in its section lower down.
- **Evidence boxes.** Each cue may carry `box` = `[x0, y0, x1, y1]` as 0-1 fractions of the upright image. The card draws them over the local preview as numbered rectangles (absolutely positioned `<span>`s in percent), numbered like the cue list. Hovering or focusing a cue highlights its box, and hovering a box highlights its cue. A checkbox hides the boxes. Link checks have no local preview, so they show the cue list only.
- **Uncertain.** An `uncertain` verdict adds a note ("We would rather say we don't know than guess") with the calibrated probability. Under the big number, "What this number means" links to `/scorecard#calibration`.
- **Download report.** The button stamps the UTC time and calls `window.print()`. Print CSS (`@media print` in `globals.css`) hides everything except the result card, so the report is the card itself: verdict, probability, boxes over the preview, cues, provenance, detector score, model version and disclaimer. Choose "Save as PDF" in the print dialog. Nothing is uploaded or stored.

## Scorecard page

`/scorecard` is rendered on the server from `GET /v1/scorecard` (see `docs/api.md`, **Scorecard**). It shows the catch rate per generator, the false-alarm rate per real-photo source (clean and compressed), clean vs compressed overall, the Claude Haiku visual check's accuracy and uncertain rate, the calibration sentence (`#calibration`) and limits. If the endpoint fails or returns 404 it says the scorecard is unavailable. It is linked from the header and from the facts line on the home page. Mock mode uses `MOCK_SCORECARD` in `web/lib/api.ts`; `MOCK_RESULT` has two boxed cues and one whole-image cue.

## Style

Plain CSS in `web/app/globals.css`, the system font stack (no web font), white background in every mode (no dark theme), colours and radii copied from avikwok.dev (green `#16a34a`, darker green-ink for text and buttons, 16px cards, 6px nested items, 4px chips). Verdict colours: AI red, real green, uncertain amber, each at least 4.5:1 on white. The "How it works" strip on the main page reads the model version and one metric from `GET /v1/model` and hides them if the call fails.
