# Data pipeline (ml/data)

For a beginner. Read top to bottom once; later use it as a reference.

## What problem does this solve?

A detector learns from examples: images labelled **real (0)** or **AI (1)**. If the examples differ in
some boring way (fakes are PNG, reals are JPEG; fakes are 1024 px, reals are 640 px), the model learns the
boring difference and fails on real-world uploads. This folder builds a dataset where that cannot happen,
then checks it with a shortcut audit.

## The files

| File | What it does |
|---|---|
| `sources.yaml` | List of every image source: real or AI, which generator, licence, notes, `enabled` on/off. Also names the held-out generator. |
| `sources.py` | Reads the registry; reads and writes each source's `index.jsonl` (one JSON line per downloaded image). |
| `download.py` | One downloader per source type. Resumable: rerun it and it skips files it already has. `--grow N` adds capped new images. |
| `generate_local.py` | Makes free AI images on your Mac (SDXL-Turbo) with COCO captions as prompts, so each fake has a real twin with the same content. `--grow N` adds a capped batch. |
| `generate_paid.py` | Placeholder for paid APIs. Refuses to run without `--approve-spend <dollars>` and has no provider wired in. |
| `normalize.py` | Makes every image look identical in format: decode, convert to RGB, resize the shorter side to 512, center-crop 512x512, save as JPEG quality 95, no metadata. |
| `splits.py` | Decides train / val / gate / report_test and holds the frozen report_test list. |
| `audit.py` | The shortcut audit. |
| `augment.py` | Train-time distortions (blur, resize, crop, JPEG) so the model survives compressed uploads. |
| `build.py` | Runs the steps above and writes `data/manifest.parquet`. |
| `frozen/report_test.txt` | The frozen list of report_test image ids. Written once by the freeze step, then never rewritten. |

Everything big lives in `data/` (git-ignored): `data/raw/<source>/` originals, `data/normalized/` the
512x512 JPEGs, `data/manifest.parquet` the table the rest of the project reads.

## Commands

```bash
make data LIMIT=50                                      # = python -m ml.data.build --limit 50 (smoke scale)
make full                                               # full scale, hours; resumable; logs in logs/ (see Full scale)
.venv/bin/python -m ml.data.build --limit N --freeze    # the one-time freeze of report_test; N is full scale (make full passes 2500)
.venv/bin/python -m ml.data.download --grow N           # add up to N new images per growable source
.venv/bin/python -m ml.data.download --source openimages_original --limit 2500   # Open Images originals: Flickr camera photos, CC BY 2.0
.venv/bin/python -m ml.data.generate_local --grow N     # add up to N new SDXL-Turbo images (needs ~7 GB of weights on first run)
.venv/bin/python -m ml.data.audit                       # rerun only the shortcut audit on the current manifest
.venv/bin/python -m pytest tests/test_data.py -q
```

`--limit N` means "at most N images per source". `--grow N` means "add up to N new images per source on top
of what is already indexed". Run `generate_local` before or after `build`, then rerun `build`; it is fast the
second time because downloads and normalization are cached.

## The manifest

`data/manifest.parquet`, one row per unique image:

| Column | Meaning |
|---|---|
| `image_id` | sha256 of the normalized JPEG bytes (so duplicate pixels collapse into one row) |
| `path` | absolute path of the normalized JPEG on the machine that built it |
| `label` | 0 real, 1 AI |
| `generator` | `real_coco`, `real_openimages`, `real_oi_original`, `real_raise`, `dalle3`, `flux`, `midjourney`, `stable_diffusion`, `sdxl_turbo` |
| `source` | registry entry name |
| `split` | `train`, `val`, `gate`, `report_test` |
| `license` | licence text of the source (Open Images rows carry the per-image licence URL) |

## Splits and the held-out generator

* **report_test**: every image from one generator the model never trains on (`flux`, set by
  `holdout_generator` in `sources.yaml`), plus the RAISE reals (raw camera files converted from Nikon NEF;
  never used for training, see [Camera photos in training](#camera-photos-in-training)).
  This is the honest score: "does it catch a generator it has never seen?"
* **train / val / gate**: everything else, split 70/15/15 by a hash of the *content group*. A COCO photo and the
  AI image made from its caption share a group, so they land in the same split and no twin leaks across.

### The frozen list

* There is one list: `ml/data/frozen/report_test.txt`. It holds the image ids of report_test, with a digest
  line at the top.
* It is written **once**, by the explicit freeze step (`--freeze`). `make full` runs that step at full scale.
  Nothing rewrites it after that.
* Later builds keep exactly those report_test images. New images from the held-out generator or from RAISE
  are dropped, so they are never trained on.
* Bigger builds only add to train/val/gate. Hash splits keep every existing row in the split it already had.
  A bigger build never raises an error.
* Check that the two numbers printed by `build` match: `frozen list: N ids` and `report_test: M rows`. A
  smaller M means a frozen image is missing from the raw data.
* **Do not freeze at smoke scale.** A frozen list made from 50 images per source would be too small to publish.
* Resetting means deleting `ml/data/frozen/report_test.txt` by hand and freezing again. This invalidates every
  number published from the old list, so only do it on purpose.

## Camera photos in training

### Why camera photos were added

- COCO and Open Images give web copies: photos that were resized, recompressed and often made small before they were shared.
- Probes trained only on those learned that "real" looks like a web copy. On photos straight from a camera they went wrong: they flagged about one in five RAISE camera photos as AI.
- The `openimages_original` source adds camera photos to train, val and gate, so the probes see camera output during training. The val split, which sets the thresholds, holds camera photos too.
- The live model (v16) was trained on 227 of them: 167 in train, 32 in val, 28 in gate. The 2500 in the table below is the target, not the current count.
- With them, RAISE photos wrongly flagged as AI went from 20.4% (v13) to 18.1% (v15) to 6.7% (v16). See [results.md](results.md).

### Where the camera photos come from

Every Open Images image was made from a Flickr photo. The validation CSV lists, for each image, the Flickr original (`OriginalURL`), its licence, its author and its Flickr page. The `oiorig` downloader:

- reads the validation CSV that the `openimages` source already uses;
- skips images already in the `openimages` index;
- keeps only CC BY 2.0 rows;
- downloads the Flickr original and keeps it only if its EXIF names a camera maker (the `Make` tag). A file with no camera make is often a scan, a screenshot or an export;
- applies the EXIF rotation, shrinks the photo to 1600 px on the long edge and saves it as JPEG quality 90;
- writes each photo's author, author page, Flickr page and licence into `data/raw/openimages_original/index.jsonl`.

The `openimages` source skips every photo already kept here, so a photo is never stored twice. Dead links and photos without a camera make are listed in `_rejected.txt` and are not fetched again. Flickr limits how fast one machine may download, so the downloader runs 2 requests at a time, waits 3 to 8 seconds before each one, and waits longer after each refusal (HTTP 429). It stops when many photos in a row keep failing; rerun it later and it resumes.

### Why RAISE is not used for training

RAISE is the real half of the frozen report_test. 300 were downloaded and 299 passed the build checks, so report_test holds 299. They come from two cameras (Nikon D7000 and D90)
and one conversion path (NEF to JPEG with macOS `sips`).

Training on other RAISE images would teach the probes the look of exactly those cameras and that conversion.
The test would then reward memory of the test cameras. It would not measure camera photos in general, and the
RAISE score would look better than the detector really is. So RAISE stays test-only. Training camera photos
come from a different source with many cameras and photographers.

### Caveat

A camera make in EXIF shows that a camera wrote the file. It does not show that the photo is unedited: some Flickr originals were edited before upload and kept their EXIF. The photos span many years of cameras and phones.

## Prompt partition

In the Rapidata dataset, every prompt has images from all four generators. To keep the test honest, prompts
are split by a hash of the prompt:

* 30% of prompts (`HOLDOUT_PROMPT_PCT` in `download.py`) are used only for the held-out generator (`flux`).
* The other 70% are used only for the training generators (DALL-E 3, Midjourney, Stable Diffusion).

So no training image shares a prompt with a test image. As a second check, `build` also drops any train, val or
gate row whose content group appears in report_test.

## Why normalize both sides identically

Real photos arrive as JPEG with camera metadata; AI images arrive as PNG/WebP, often with no metadata. The
normalizer erases that: same size, same JPEG quality and quantization tables, no EXIF/XMP/C2PA. Caveat: a real
JPEG that was already compressed gets compressed twice, an AI PNG only once. Double-compression leaves faint traces
that file-size tests cannot see; the augmentation step (random JPEG quality on training copies) is there to
keep the detector from leaning on it.

## The shortcut audit

The audit asks one question: can plain facts about the files, with no picture content, tell real from AI?
It fits a small logistic regression on those facts and reports a cross-validated AUROC. Two kinds of facts are
handled differently.

**Gated (the build fails if these leak):** width, height, file format, JPEG quantization tables and metadata
presence. These are things the pipeline controls. If they differ between real and AI, the normalizer failed,
and a detector would learn the file instead of the picture. After normalization they are the same for almost
every image, so the gated AUROC sits at 0.5 (a coin flip). The limit is 0.6. Metadata presence is the one
gated feature that still varies: at smoke scale one of 335 images still carries a JPEG comment segment.

**Info only (never gated): file size.** After identical re-encoding, file size is set by how much fine detail
a picture has. AI images often have smoother textures than camera photos, so their files come out smaller.
That is texture, the same kind of signal the detector is meant to learn. It is not a pipeline leak. Gating on it
would fail every honest dataset, because real photos and AI images differ in texture in any honest sample. So the
audit prints it, and the build does not gate on it.

The build exits 0 when the gated features pass and exits 1 only on a real leak.

Reading the numbers: AUROC 0.5 means the facts cannot separate the classes; 1.0 means they separate them
perfectly. For file size, **below 0.5 means AI files are smaller**, above 0.5 means AI files are larger.

Current numbers, from `.venv/bin/python -m ml.data.audit` on the smoke manifest (n=335):

```
shortcut audit: AUROC=0.500 (limit 0.6) on n=335
  single width: 0.500
  single height: 0.500
  single is_jpeg: 0.500
  single q_luma_mean: 0.500
  single q_chroma_mean: 0.500
  single has_metadata: 0.496
  info (not gated): file_size AUROC 0.276
PASS
```

So the gated AUROC is 0.500, and file size is 0.276 (AI files are smaller). Smoke size is small, so these
numbers will move at full scale.

## Augmentation

`augment(img, seed)` applies, in order: 30% chance of light Gaussian blur, resize by 0.5 to 1.5, random crop
(70-100% of each side), JPEG re-encode at quality 30-95. Training images get 2 augmented copies (`N_AUG_COPIES`)
besides the clean one. The same seed always gives the same output, so cached features stay valid.
Never augment val, gate or report_test.

## Full scale

`make full` runs the whole chain: download, generate, build with the freeze step, extract features, train and
eval. It is resumable: after an interruption, rerun `make full` and finished work is skipped. Logs go to
`logs/full-<stage>.log`. The default cap is `FULL_LIMIT=2500` images per source (`make full FULL_LIMIT=500`
for a smaller run). The whole run takes hours.

| Source | Side | Generator | Target count | Notes |
|---|---|---|---|---|
| COCO 2017 val | real | `real_coco` | 2500 | seeded sample; captions drive local generation |
| Open Images validation | real | `real_openimages` | 2500 | CC BY 2.0 images |
| Open Images originals | real | `real_oi_original` | 2500 | Flickr originals with a camera make in EXIF, CC BY 2.0; train/val/gate only; 273 downloaded so far, 227 used in v16 |
| RAISE | real | `real_raise` | 300 | test only (report_test); NEF converted with macOS `sips` |
| SDXL-Turbo (local) | AI | `sdxl_turbo` | 2500 | one image per COCO caption |
| Rapidata | AI | `dalle3` | ~760 | from the Rapidata shard lists |
| Rapidata | AI | `midjourney` | ~770 | from the Rapidata shard lists |
| Rapidata | AI | `stable_diffusion` | ~790 | from the Rapidata shard lists |
| Rapidata | AI | `flux` | ~330 | held out; test only (report_test) |

Totals: about 7.8k real (2500 COCO + 2500 Open Images + 2500 Open Images originals + the 299 RAISE test images)
and about 5.1k AI across 5 generators. These are
the counts in the top comment of `sources.yaml`. Rapidata counts are estimates from the shard listing, so the
real numbers may differ a little.

Only sources that download without a login are enabled. Surveyed sources that are too large or need a
different reader are listed in `sources.yaml` with `enabled: false`.

## Nightly growth

After the frozen list exists, new images arrive each night in capped batches: new downloads and new local
SDXL-Turbo images, never the held-out generator and never RAISE. Growth only adds to train/val/gate, so the
frozen report_test stays fixed. See [loop.md](loop.md) for the nightly steps and what they never touch.

## Sources: what is used, what was surveyed

See `sources.yaml` for the full list with licences. In short:

* **Used**: COCO 2017 val (reals + captions), Open Images validation (reals, CC BY 2.0), Open Images originals (Flickr camera
  photos for train/val/gate, CC BY 2.0), Wikimedia Commons (camera photos for train/val/gate; CC0, public domain,
  CC BY or CC BY-SA per file), RAISE (raw camera reals; licence is non-commercial research
  only), Rapidata Flux/SD/Midjourney/DALL-E 3 comparison dataset on
  Hugging Face (AI, `cdla-permissive-2.0` per its HF tag), local SDXL-Turbo.
* **Surveyed, not enabled**: GenImage (620 GB, 2022-23 generators), Synthbuster (12 GB zip, CC BY-NC-SA),
  Photoroom Midjourney-v6 recap, OpenDatasets DALL-E 3, FLUX-Reason-6M. All open without login; all are too big
  for a smoke run. Candidates for later scale-ups.

### Wikimedia Commons camera photos

Source `wikimedia_commons`, generator `real_commons`. It adds a second set of camera photos to train, val and gate.
It is never used for report_test. Run it on its own with:

```bash
.venv/bin/python -m ml.data.download --source wikimedia_commons --limit 2000
```

How the `commons` downloader picks photos:

1. It lists every "Taken with <camera>" category with at least 20 files. Commons sorts photos into these
   categories by camera model: DSLRs, mirrorless cameras, compacts and phones. The list is saved in
   `data/raw/wikimedia_commons/_categories.txt`, so a rerun walks the same categories in the same order.
2. It walks the categories in a seeded random order and keeps at most 3 photos per category, so no single
   camera dominates.
3. For each candidate it asks the API for the image size, the licence and the EXIF tags. A file is kept only if
   all of these are true:
   - it is a JPEG and its short side is at least 1000 px;
   - its licence (`LicenseShortName`) is CC0, public domain, CC BY or CC BY-SA;
   - EXIF names the camera make and model;
   - the EXIF date taken (`DateTimeOriginal`) is before 2022-01-01. Photo-realistic AI images became common
     after that, so this guards against AI images with copied EXIF;
   - none of its categories mentions AI-generated, Midjourney, Stable Diffusion or DALL-E.
4. Only files that pass are downloaded. The original is turned upright with its EXIF rotation, shrunk to
   1600 px on the long edge and saved as JPEG quality 90.
5. Each kept photo gets a line in `index.jsonl` with its title, author, licence name, licence URL, file page
   URL, camera make and model, date taken and camera category.
6. Rejected files go to `_rejected.txt` with the reason (`small`, `not_jpeg`, `licence`, `no_camera`, `date`,
   `ai_category`, `dead`, `missing`). They are never checked again.

It follows the Wikimedia rules for bots: a `User-Agent` with a contact link, one API request at a time with
`maxlag=5`, a pause after each API request, at most 2 downloads at once under 25 Mbps in total, a random
2 to 5 second pause before each download, and the server's `Retry-After` wait after a refusal. If several
downloads in a row fail every retry, it stops. Rerun it later and it resumes where it stopped.

Caveat: many Commons photographers edit their photos (for example in Lightroom) before upload. EXIF shows that a
camera took the photo, not that the file is unedited.

**Licensing notes**:

* **SDXL-Turbo**: Stability AI Community License (last updated July 5, 2024). Research, non-commercial and
  limited commercial use are free. Commercial use needs registration with Stability AI and annual revenue under
  US $1M.
* **RAISE**: scientific non-commercial use only. Used for test only, never trained on.
* **Open Images and Open Images originals**: each image is under CC BY 2.0 (the `License` column of the Open Images validation CSV); the manifest's `license` column holds that licence URL. For the originals, the author, author page, Flickr page and licence of every photo are kept in `data/raw/openimages_original/index.jsonl`. Credit: the Open Images dataset and the Flickr photographers named there. The images are not redistributed.
* **Wikimedia Commons**: the licence is set per file, and only CC0, public domain, CC BY and CC BY-SA files are kept.
  The author, licence and file page of every photo are kept in `data/raw/wikimedia_commons/index.jsonl`. CC BY and
  CC BY-SA need credit to the authors named there. CC BY-SA also asks that adaptations be shared under the same
  licence; whether a trained model counts as an adaptation is legally unsettled, so check this before publishing
  model weights. The images are not redistributed.
* The images never go in git. Decide the licence question before any public release.

## Beginner glossary

* **AUROC**: probability that a random AI image scores higher than a random real one. 0.5 = guessing.
* **Held-out generator**: an AI model whose images are used only for the final test.
* **Content-matched**: the fake was made from a description of the real photo, so topic and composition match.
* **Resumable**: rerun after a crash and it continues instead of starting over.
