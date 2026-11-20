# Video checks, explained for beginners

A video check asks the same question as an image check: does this look fully AI-generated? A video is harder to judge than a picture, so the server works with still frames pulled out of it. This page follows one video from upload to answer. The route is in `api/main.py` (`check_video`), and the video code is in `api/video.py`. The request and response formats are in [api.md](api.md).

Video checks need `ffmpeg` and `ffprobe` installed on the server. See the settings in [api.md](api.md).

## What happens to one video

1. **Upload.** The file goes in the form field `video`. We read the first bytes to see if it is an MP4, WebM or MOV file, because file names lie. The upload is copied in 1 MB chunks into a temporary folder. Over 100 MB gives `413`. An empty upload gives `400`. Any other type gives `415`.
2. **Probe.** `ffprobe` reads the length and the frame size. Longer than 60 seconds gives `413`, and so do frames over 40 megapixels. A file `ffprobe` cannot read gives `400`.
3. **Frames.** `ffmpeg` pulls up to 20 still frames out of the video: 16 evenly spaced, plus up to 4 at scene cuts (see the next section).
4. **Detector.** Each frame gets a detector score. Videos use two of the three image backbones, in half precision (see below).
5. **Judge.** The server puts the frames into one picture of two labelled grids. Claude Haiku 5.5 reads that picture while the frames are being scored.
6. **Answer.** Claude's verdict is used when it is available. If not, the detector band is used, and the response says why.

The temporary folder is deleted when the request ends, whether it worked or not. Nothing is stored.

## Frame sampling

- We pick 16 timestamps. Each one is the centre of one of 16 equal slices of the video. An 8 second video has slices of 0.5 seconds, so the timestamps are 0.25, 0.75, 1.25 and so on up to 7.75.
- Why the centres? Many videos start with a black or faded frame, and seeking to the very last moment often fails. The centres avoid both.
- `ffmpeg` runs once per timestamp. Each run seeks to that time, grabs one frame, and scales it to at most 1280 px on each side.
- If a frame is identical to the one before it, we drop it. Very short clips can seek to the same frame twice, so a short clip may give fewer than 16 frames.
- After the 16 even frames, one more `ffmpeg` pass decodes the video at a small size and lists the hard cuts (frames whose picture differs from the one before by more than 40 %). Up to 4 cuts become extra frames, so a video never has more than 20. A cut is skipped when it is within 0.5 seconds of a frame we already have, so short videos usually get none. If this pass fails or runs out of time, we use the even frames alone.
- All frame extraction stops after 30 seconds in total. Any frames taken before that point are used.

## Aggregation: from frame scores to one video score

Each frame gets an ensemble score, like an image (see [detector.md](detector.md)), but from two backbones instead of three. The video score is the **mean of the 4 highest frame scores**.

Why the top 4 and not the average of all frames?

- A video often has a few frames that look clearly generated and many that look ordinary. An average over all frames dilutes the clear ones.
- The single highest frame is noisy. One odd frame should not decide the answer. Four frames give a steadier number.
- If fewer than 4 frames survive sampling, the mean covers the frames that exist.

Example with made-up scores: 0.10, 0.20, 0.90, 0.95, 0.85, 0.80, 0.30 and the rest lower. The top four are 0.95, 0.90, 0.85 and 0.80, so the video score is 0.875.

Two thresholds turn this score into a band: `likely_ai`, `uncertain` or `likely_real`. Videos have their own thresholds. A two-backbone score is not on quite the same scale as a three-backbone one, and real video scores higher on the image probes than real photos do (see Results). They live in `video_meta.json` next to the model's `meta.json`, and for model v16 they were picked on the video validation set (`t_ai` = 0.974, `t_real` = 0.038). If that file is missing, the image thresholds are used, and for video that is a bad fit: it flagged about 65 % of real test videos (see Results). So every model that goes live needs a `video_meta.json`.

## Speed: two backbones in half precision

A video has up to 20 frames, and each frame goes through the detector. With all three image backbones, that was the slowest part of a video check. So videos are scored differently from images:

- **Two backbones, not three.** Videos use CLIP and SigLIP. DINOv2 is skipped. It is the weakest of the three on its own, and it took about half the scoring time.
- **Half precision on the GPU.** On a Mac GPU (MPS), the two backbones run in 16-bit floats instead of 32-bit. This is faster and uses less memory. On a CPU they stay in 32-bit, because 16-bit math is slow there. So on a CPU server, videos reuse the backbones already loaded for images instead of loading a second copy, which saves about 2.5 GB of memory.
- **Images do not change.** An image check still uses all three backbones in 32-bit.
- **Video thresholds.** The video thresholds are picked on the validation set of videos (not on still images) with the same rule the image thresholds use. The held-out test set is only used to report the numbers, never to pick anything.

What we measured on still images (model v15, kept as history; the speed-up and half-precision checks below do not depend on the model version):

| | 3 backbones (image setup) | 2 backbones (video setup) |
|---|---|---|
| AUROC on the held-out test set | 0.981 | 0.978 |
| Share of AI images caught when 5 % of real photos are flagged | 0.918 | 0.929 |
| Image-picked thresholds `t_ai` = `t_real` (v15, history) | 0.227 | 0.251 |

- **Speed.** On a 16-frame test clip on an M2 Mac, scoring the frames took 5.5 seconds with the video setup and 14.0 seconds with all three backbones in 32-bit. That is about 2.5 times faster. Other jobs were running on the machine at the same time, so both numbers are slower than on an idle machine. The ratio is the useful part.
- **Half precision barely moves the scores.** On 64 validation images (32 real, 32 AI), the largest change in a backbone score from 32-bit to 16-bit was 0.004. The largest change in the final score was 0.0025. No image changed band.

The video thresholds are stored in `models/vN/video_meta.json`: the backbones, `t_ai`, `t_real`, the split they were picked on (`val`), and how many items that split had. For v16 that is 1,200 validation videos. (v15's file was picked on still images and says 1,461.)

## Why Claude sees frame grids

Claude accepts images and PDFs. It does not accept video or audio. So the server does not send the video at all. It builds one picture from the frames:

- **Grids of 8.** Frames 1 to 8 are in the first grid, frames 9 to 16 in the second, and any scene-cut frames after that in a third. All grids sit in one JPEG. Our judge backends send one image per call, so the whole video costs one call.
- **Labels.** Each frame has a label with its number (`#1` to `#20`) and its time in seconds, so Claude can refer to a frame by number.
- **Size.** The picture is at most 1568 px on its longest side. The API may downscale the picture again, so tiles can end up slightly smaller.
- **Time order.** The prompt tells Claude the frames are still frames in time order. Claude is not told the detector scores.
- **Frame numbers, not times.** Claude names the frame for each cue. The server looks up the time from that frame number. Times written by the model are never trusted, and a cue that names a frame that does not exist is dropped.
- **Parallel work.** The judge runs while the detector scores the frames, so the wait is about the slower of the two steps.

Claude judges what it can see in still frames. It cannot see motion between frames directly. The temporal features planned for the full v4 are meant to cover that gap.

## Safety

- The upload is given to `ffmpeg` only as a file path inside our own temporary folder. Nothing is passed as a URL or a pipe.
- The input format is set from the bytes we sniffed, not from the file name, so `ffmpeg` does not have to guess.
- Only the `file` and `pipe` protocols are allowed. A crafted playlist (HLS or concat) that points at `http://` is refused, so `ffmpeg` never opens a network connection. A test in `tests/api/test_video_ffmpeg.py` proves this with a local socket that must never be contacted.
- `ffmpeg` runs with `-nostdin`, so it never waits for input, and with a 30 second limit. It uses at most 2 threads.
- Nothing is stored. The temporary folder is always deleted. The web framework may first put an upload over 1 MB into its own anonymous temporary file while it reads the request. We do not keep that file.

## Limits

- Size: 100 MB. Length: 60 seconds, with half a second of allowance for container rounding. Frame size: 40 megapixels.
- Formats: MP4, WebM and MOV only.
- Rate limit: a video counts as 3 of the 10 checks per IP per hour.
- Judge: one call per video.
- Frames only for the verdict: there is no motion analysis. The soundtrack is scored only when the prototype audio check is switched on (off by default, see [api.md](api.md)); its result sits next to the picture verdict and feeds the overall verdict (next section). When it is off, the response says audio was not analysed in `audio.note`.
- WebM files that do not record their length are refused with `400`, because the frame times depend on the length.
- Frame sampling stops after 30 seconds.

## What v4 added after the prototype

- **Real training data.** AI-generated and real videos from AIGVDBench, used to train and test a video probe (next section).
- **Temporal features.** Four motion numbers inside that probe: how much the picture changes from frame to frame. The probe did not beat the frame rule, so videos are still scored with the frame rule.
- **Audio.** A voice-deepfake probe on XLS-R speech embeddings scores the soundtrack when audio checks are switched on and the soundtrack has clear speech (see [audio.md](audio.md)). Music or noise alone gets no voice result. Its result is reported next to the picture verdict and also feeds the overall verdict (next section).

## Overall verdict

A video result shows three things: the picture verdict (from the frames), the voice result (only when audio checks are on and the video has a soundtrack with clear speech; music or noise alone gets the note "No clear speech found, so the voice was not checked." and no band), and an overall verdict. The overall rule is simple and is not trained:

- If the picture or the voice says "likely AI", the overall verdict is "likely AI". The page says which part flagged it.
- If the picture says "likely real" and the voice (when there is one) says "likely real", the overall verdict is "likely real".
- Anything else is "uncertain".
- With no voice result (audio checks off, no soundtrack, no clear speech, or an unreadable track), the overall verdict is the picture verdict.

Flagging on either part raises false alarms, because a real video only has to fool the rule once. The picture verdict shown is the judge's when it answers. On 150 real test videos the judge wrongly flagged 1.3% (2 videos); see "How often the video verdict is right" below. When the judge is unavailable, the detector is used instead, and it wrongly flagged 5.3% of real test videos. The voice check wrongly flagged 3.3% of real speech on In-the-Wild. Before the speech gate, it also called 54.9% of 390 real music and ambient clips `likely_ai`; with the gate, 2.3% (see [audio.md](audio.md) section 8). So with the judge, roughly 1 to 5% of real videos can be flagged overall, and up to about 9% when only the detector is available. The voice check is a prototype (see [audio.md](audio.md)). The combined rule itself has not been measured on videos with real soundtracks. In the API the fields are `overall_verdict` and `overall_flagged_by`.

## How often the video verdict is right

We ran the production path (the same frame grid, prompt and judge, Claude Haiku 5.5) on 300 test videos picked at random before any result was seen: 150 real and 150 AI from Gen2, Gen3, Sora and Kling. Three videos were skipped because the API would refuse them (two over 60 seconds, one unreadable), so 297 were scored. The judge answered every time. "Accuracy" counts "uncertain" as wrong. Ranges are 95% bootstrap intervals.

| | Accuracy | AI videos caught | Real videos wrongly flagged | Uncertain |
|---|---|---|---|---|
| Judge (what you see) | 72.7% (67.7 to 77.8) | 68.7% (60.9 to 75.9) | 1.3% (0.0 to 3.4) | 20.2% |
| Detector alone (v16 video cut-offs) | 35.7% | 56.5% | 4.7% | 61.6% |

AI videos caught, per generator:

| Generator | Judge | Detector |
|---|---|---|
| Gen2 | 82% | 84% |
| Gen3 | 69% | 49% |
| Sora | 45% | 47% |
| Kling | 81% | 44% |

What this means:

- **A "likely AI" verdict is strong evidence.** Only 2 of 150 real videos got one.
- **A "likely real" verdict is weak evidence.** About 1 in 3 AI videos was not caught, and the judge called 19 of 147 AI videos real. Sora is the hardest: fewer than half were caught.
- **"Uncertain" is common.** The judge said it for 1 in 5 videos. The detector alone is undecided on most videos, which is why the judge is shown first.
- The judge's confidence number is its own estimate, not calibrated on video. Each call took about 8 seconds.
- These are the picture part only. The voice check was not part of this test, and all real test videos come from one dataset, so results on phone or social-media video may differ.

## Trained video probe

The image detector was trained on still images only. This section asks two questions: how well does the frame rule above work on real AI videos, and does a small model trained on videos do better? The code is in `ml/video/`. The trained probe is an experiment. The API does not use it yet.

### Data and licences

We use [AIGVDBench](https://huggingface.co/datasets/AIGVDBench/AIGVDBench) (2026). It is open: no account, no gated terms.

- **Licence.** The dataset card says CC BY 4.0. Its real videos come from OpenVid-1M, which is "intended for research and non-commercial purposes". It also holds outputs of closed services (Sora, Kling, Runway). So we treat all of it as **research and non-commercial use only**. This project is non-commercial, so that is fine here. A commercial version would need other training data.
- **Download.** The dataset has one zip per generator, up to 94 GB each. We never download a whole zip. `ml.video.data` reads each zip's table of contents with HTTP range requests, then fetches only the chosen videos (one range request each) and checks each file's CRC. The subset is 11,208 videos, 40 GB, in `data/raw/video/aigvdbench/` with a `manifest.csv`.
- **Split by generator.** A detector is only useful if it catches generators it has never seen, so the split is by generator, never by video. Real videos follow the dataset's own train/val/test ids, so no clip is in two splits. The ids are clip ids, though, and 199 source videos have clips in more than one split. Among the scored videos, 31 source videos have clips in both train and val, 13 in both train and test, and 13 in both val and test (8, 6 and 10 when only real clips are counted). That is 1 to 3 % of a split, too little to move the numbers much.
  - **train:** AI videos from 16 open generators, about 188 each: EasyAnimate (I2V), LTX (I2V), SVD (I2V), VideoCrafter (I2V), AccVideo (T2V), AnimateDiff (T2V), Cogvideox1.5 (T2V), EasyAnimate (T2V), IPOC (T2V), LTX (T2V), Open-Sora (T2V), RepVideo (T2V), VideoCrafter (T2V), Wan2.1 (T2V), Cogvideox1.5 (V2V), LTX (V2V).
  - **val:** 3 open generator families (4 generators) held out from training, 150 videos per generator: Pyramid-Flow (I2V), SEINE (I2V), HunyuanVideo (T2V), Pyramid-Flow (T2V). Val picks every setting (C, temperature, thresholds).
  - **test:** 4 closed models, 500 videos each: Gen2, Gen3, Sora, kling. Test never picks anything. It was scored three times, always with the same rules: in a small dry run (99 videos), on a first part of the data (792 videos), and on the data used below (1,189 videos).
- **What was scored.** Pulling frames and turning them into embeddings are the slow steps (embedding runs at about 5 frames per second on the Mac GPU, which other jobs also share). Each split was shuffled once with a fixed seed into shards of 100 videos, and only whole shards were used, so each scored part is a fair random sample. Frames were pulled for 7,200 videos: the first 20 train shards (2,000 of the 6,008 downloaded train videos), all 12 val shards (1,200) and all 40 test shards (4,000). Every train and val video with frames was scored, but only the first 12 test shards. Scored: train 2,000 videos (1,032 AI, 968 real), val 1,200 (600 AI, 600 real), test 1,189 (605 AI, 584 real). `models/video/m1/meta.json` does not list the shards, and `ml.video.probe` loads every shard that exists, so rerunning it after more shards are embedded gives different numbers.
- **Refused videos.** A video the API would refuse is skipped, as the API would skip it. Of the 7,200 videos we pulled frames from, 34 were refused, all of them test videos: 29 longer than 60 seconds, 3 unreadable, 2 larger than 100 MB. 11 of them fall in the 12 scored test shards, which is why test has 1,189 videos and not 1,200.

### Method

**Frames.** For each video we take the same frames as the API: 16 evenly spaced, plus up to 4 at scene cuts, scaled to at most 1280 px. Then each frame gets the training normalizer (512 px centre crop, JPEG quality 95), as the API does. The API seeks once per frame, which is fine for one upload but too slow for 11,000 videos. So the batch code (`ml.video.extract.fast_frames`) decodes each video once and keeps the frames at the same times. It uses the Mac's hardware video decoder for plain 8-bit video. Videos shorter than 16 seconds skip the scene-cut pass, because no cut can be added to them (every moment is within 0.5 s of an even frame). We checked the result against the API's own code on random downloaded videos: all frames the API produced were byte-identical (120 of 120 on 12 videos in the saved log, 214 of 214 on 20 videos in a re-run). On real videos the API always hit its 30-second budget before the scene-cut step, so the cut frames were compared only on a synthetic clip with a hard cut. On a busy machine the API's 30-second budget sometimes stopped after fewer frames; the batch code has no budget, so it always has the full set. Tests (`tests/test_video_probe_frames.py`) check the match on a 3-second clip and on an 18-second clip with one hard cut.

**Embeddings.** Each frame goes through CLIP ViT-L/14 and SigLIP, the two video backbones of the API. On the Mac GPU they run in 16-bit floats, as in the API. Because other jobs held the GPU for hours, part of the data was embedded on the CPU in 32-bit floats, which is what the API does on a CPU server. Earlier we measured that 16-bit versus 32-bit moves a frame score by at most 0.0025. Within each shard of 100 videos, AI and real videos went through the same device, so the device cannot leak the label. Each shard file records which device made it. In this run: train on the CPU, val on the GPU, test on the GPU except 3 of its 12 shards.

**Baseline: the frame rule.** Exactly what the API does now. Each frame gets the image ensemble score from the CLIP and SigLIP probes of the live model (v16 here; the v15 numbers are kept as history). The video score is the mean of the 4 highest frame scores. The band uses `t_ai` and `t_real` from `video_meta.json`. v16 had no such file, so live video used the image `t_ai` of 0.167, which is the "before" row below. v15's file held thresholds picked on still images.

**Video probe m1.** One logistic regression per video, on 1,542 numbers:

- the average CLIP embedding over the frames (768) and the average SigLIP embedding (768);
- the highest frame-probe logit for each backbone (2);
- 4 motion numbers: for each backbone, the mean and the spread (standard deviation) of how much the embedding changes from one frame to the next (cosine distance). A real camera moves smoothly; a generator can drift or flicker.

The features are standardized with the train mean and spread. C (how strongly the weights are held back) was chosen from 0.001 to 1 by val AUROC: C = 0.01. Then a temperature (0.76) was fit on val so the output reads as a probability. Thresholds use the image rule on val: `t_ai` is set so that at most 5 % of val real videos are flagged, `t_real` so that at most 10 % of val AI videos are called real. For m1, `t_ai` = 0.666 and `t_real` = 0.663. The two are almost equal, so on val the probe leaves almost no video in the "uncertain" band.

Files: `models/video/m1/probe.json` (the weights, in the same format as the image probes) and `models/video/m1/meta.json` (thresholds, settings and every number below). Rebuild with:

```
python -m ml.video.data                  # download the subset (40 GB)
python -m ml.video.extract frames        # CPU: frames for every video (we stopped train at --limit 2000)
python -m ml.video.extract embed         # GPU: embeddings (--device cpu works too, slower); stops at the first shard without frames
                                         # (the numbers below used train shards 0-19, val 0-11, test 0-11, e.g. --splits test --shards 12)
python -m ml.video.probe                 # baseline + probe; writes models/video/m1
```

### Results

How to read the columns:

- **AUROC**: the chance that a random AI video scores higher than a random real one. 0.5 is a coin flip, 1.0 is perfect. It does not depend on any threshold.
- **AI caught at 5 % false alarms**: set the threshold so that 5 % of the test's real videos are flagged, then count the AI videos above it. This uses the test reals to place the threshold, so it is a best case, not a number you get in use.
- **At `t_ai`**: the real band rule, with the threshold fixed before the test was seen.

| Method | Val AUROC | Test AUROC | Test: AI caught at 5 % false alarms | Test: AI flagged at `t_ai` | Test: real videos wrongly flagged at `t_ai` |
|---|---|---|---|---|---|
| **v16 frame rule, before the fix** (no `video_meta.json`, image `t_ai` = 0.167) | 0.600 | 0.900 | 65 % | 98 % | 65 % |
| **v16 frame rule, now** (`t_ai` = 0.974 picked on val videos) | 0.600 | 0.900 | 65 % | 65 % | 5 % |
| v15 frame rule, shipped thresholds (history, `t_ai` = 0.251) | 0.595 | 0.907 | 62 % | 98 % | 67 % |
| v15 frame rule, re-picked on val videos (history, `t_ai` = 0.984) | 0.595 | 0.907 | 62 % | 67 % | 7 % |
| Video probe m1 | 0.977 | 0.837 | 43 % | 40 % | 4 % |

On the val videos at the new v16 `t_ai`: 5.0 % of real videos are flagged (that is how the threshold is picked) and 13 % of AI videos are caught. The val generators are open models that the image probes barely separate from real video, so val AI is caught far less often than test AI.

Per closed model (test). The AI videos of each model are compared with all 584 real test videos.

| Test generator | AI videos | v16 frame rule AUROC | Probe AUROC | v16 caught before (image `t_ai`) | v16 caught now (`t_ai` = 0.974) | v15 caught (history, re-picked `t_ai`) | Probe caught (its `t_ai`) |
|---|---|---|---|---|---|---|---|
| Gen2 | 155 | 0.952 | 0.862 | 99 % | 81 % | 85 % | 37 % |
| Gen3 | 150 | 0.921 | 0.766 | 99 % | 65 % | 71 % | 27 % |
| Sora | 154 | 0.866 | 0.883 | 96 % | 51 % | 51 % | 48 % |
| kling | 146 | 0.860 | 0.835 | 96 % | 62 % | 62 % | 49 % |

Per held-out open model (val), each against all 600 real val videos (v16; v15 gave 0.550, 0.469, 0.734, 0.626):

| Val generator | AI videos | v16 frame rule AUROC | Probe AUROC |
|---|---|---|---|
| I2V/Pyramid-Flow | 150 | 0.549 | 0.983 |
| I2V/SEINE | 150 | 0.510 | 0.984 |
| T2V/HunyuanVideo | 150 | 0.725 | 0.945 |
| T2V/Pyramid-Flow | 150 | 0.616 | 0.998 |

How much to trust the differences (these bootstrap numbers were made with the v15 probes and not redone for v16, whose test AUROC of 0.900 is close to v15's 0.907): we resampled the test videos 1,000 times (bootstrap, AI and real resampled separately). The 95 % range of the test AUROC is 0.891 to 0.922 for the frame rule and 0.813 to 0.861 for the probe, and the frame rule's lead is 0.042 to 0.098. Per closed model the ranges are wider: the frame rule is clearly ahead on Gen2 and Gen3, while on Sora (lead -0.052 to 0.030) and Kling (-0.033 to 0.084) the two cannot be told apart.

An earlier run on a smaller part of the data (train 400, val 300, test 792 videos) gave the same picture: frame rule test AUROC 0.892, probe val AUROC 0.942 and test AUROC 0.819.

What this shows:

- **The frame rule ranks videos well, but its threshold is wrong for video.** On the closed models its AUROC is 0.900 with v16 (0.907 with v15): a Sora or Gen-3 video usually scores higher than a real one. But `t_ai` in `video_meta.json` was picked on still images, and on video it flags 65 % of real test videos (64 % on val) with v16's image `t_ai` of 0.167, and 67 % with v15's file (65 % on val). Real camera video looks "more AI" to the image probes than real photos do, probably because of motion blur, video compression and smooth, low-noise frames.
- **Re-picking the threshold on video fixes most of that.** With the same rule and `t_ai` = 0.974 picked on the val videos (v16), 5 % of real test videos are flagged and 65 % of the AI videos are caught (v15: `t_ai` = 0.984, 7 % and 67 %). Sora is the hardest: only about half its videos are caught. That is a threshold change in `video_meta.json`, not a new model. It is now applied: `models/v16/video_meta.json` holds it. It should still be checked on more real video sources (see Limits), because every real video here comes from one dataset.
- **The video probe does not help on new generators.** It is much better on the held-out open generators (val AUROC 0.977 against 0.595) but worse on the closed models (test AUROC 0.837 against 0.907). It learned what open-generator clips look like (small, short, upscaled; see Limits), and that does not carry over well to Sora, Kling or Runway. Five times more training videos raised its val AUROC (0.942 to 0.977) and its test AUROC (0.819 to 0.837), but the gap to the frame rule on test stayed. Only Sora and Kling are about even.
- **Decision:** keep the frame rule; do not ship m1. The video-specific `t_ai` is done for v16. The next useful steps are real videos from more sources (phones, social media), and training data whose AI and real clips have the same size and length.

### Limits

- **Part of the data.** Train uses 2,000 of the 6,008 downloaded train videos, and test 1,189 of the 4,000 test videos with frames (see above), because frames and embeddings are slow on a shared Mac GPU. The rest is downloaded and ready. Val is complete but still small: a 5 % threshold on its 600 real videos rests on the top 30 of them, so the picked `t_ai` rests on few videos and is shipped as a first estimate, to be re-checked on more real video. With about 600 test videos per class, a test AUROC moves by about ±0.015 to 0.025 from sampling alone, and a per-model AUROC by about ±0.015 to 0.045 (the bootstrap ranges above).
- **One real source.** All real videos are OpenVid-1M clips: well-shot, high-resolution, stock-style footage. Phone videos, screen recordings and heavily compressed social-media re-uploads are not in it, so the false-alarm rate on those is unknown.
- **Format and length differences.** In AIGVDBench the real clips are 1920x1080 and 3 to 60 seconds long. The open-generator clips are small (336x192 to 960x544) and short (1.4 to 5 seconds). The closed models are larger again: Sora mostly 854x480 and 5 seconds, Gen-2 mostly 1408x768, Gen-3 mostly 1280x768 and about 10.7 seconds, Kling up to 1080x1920. The 512 px normalizer hides some of this, but a probe trained on open generators can still learn "small, upscaled, short clip means AI". Short clips also have their 16 frames closer together in time, so the motion numbers partly measure clip length. This is the most likely reason the probe does better on val (open generators, small clips) than on test (closed models, larger clips).
- **Closed models only in test.** Training uses open generators only. That is the point (it tests new generators), but it also means the probe has never seen the style of the newest commercial tools.
- **Not the API's speed setup.** The batch frame sampler has no 30-second budget. On a slow or busy server the API may score fewer frames than here. On videos of 16 seconds or more with long still stretches, the batch sampler can also keep one frame fewer or more than the API, because the two remove repeated frames in a slightly different order.
- **Research licence.** The probe weights come from non-commercial data. They must not ship in a commercial product.
