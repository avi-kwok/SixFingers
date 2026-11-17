# Web front end (beginner guide)

The site in `web/` is a thin client. It never analyses images itself. It sends the file to `POST /v1/check` and draws the JSON that comes back.

Flow: pick, drop or paste an image, or paste an https image link into the "or paste an image link" field next to the upload control and press Check (pasting plain text that is an https URL anywhere on the page fills that field). The link is sent as the form field `url` and the server fetches it. Then the browser checks type and size (10 MB), shows a preview and uploads with a progress bar. The result card shows the verdict band, then "X% likely AI" (`probability_ai`) as the main number with "Detector score: Y%" smaller beneath it, then the explanation and cues, a "Detector only" tag when the explainer was unavailable, a note when it overrode the detector (for example "The detector leaned AI, but Claude's visual check found it real."), a C2PA badge, the scope note and the disclaimer.

Errors map to HTTP status: 400 for a link ("That link can't be used.", or "Couldn't fetch that image. The link may have expired or not be public."), 504 the link timed out, 502 the link could not be fetched, 429 rate limit, 413 too big, 415 wrong type, 422 the image cannot be analysed ("We can't analyse this image."). A line under the upload control says uploading means agreeing to the Terms. `/terms` is a short acceptable-use page, linked in the footer next to Privacy. A spent daily budget arrives as `judge_unavailable_reason: "budget"` on a normal 200 response.

The upload goes direct to the API because Vercel limits request bodies to about 4.5 MB. Set `NEXT_PUBLIC_API_URL` for the target API, or `NEXT_PUBLIC_MOCK=1` to try the UI with no server. Details: `web/README.md`.

Keep `web/lib/types.ts` in step with `ml/common/api_models.py`.

The About page explains the two percentages in plain words.
