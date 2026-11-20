# Audio deepfake detection (beginner guide)

This part of SixFingers answers one question about a voice recording: does it sound like real human speech, or like speech made by a computer (text-to-speech or voice conversion)? Like the image detector, it gives a probability, never a certainty.

**Status:** these are trained models (`a1`, `a2`, and the best so far, `a3`) and their measurements. The API can run `a3` as a prototype that is off by default (see section 7). While it is off, which is the default, nothing a user uploads is checked for fake speech.

It reuses the image detector's recipe (see `docs/detector.md`): a large frozen model turns each clip into a list of numbers, and a small logistic "probe" learns which numbers mean "fake".

## 1. Data

| Set | What it is | Used for | Licence |
|---|---|---|---|
| ASVspoof 2019 LA, train | 25,380 clips (2,580 real, 22,800 fake). Fakes from 6 speech synthesis / voice conversion systems (A01 to A06) | training the probe | ODC-By 1.0 |
| ASVspoof 2019 LA, dev | 24,844 clips, same 6 systems, different speakers | choosing (a1: alone; a1s, a2 and a3: together with the sets below) | ODC-By 1.0 |
| ASVspoof 2019 LA, eval | 71,237 clips, 13 systems (A07 to A19). 11 are new; A16 and A19 reuse A04 and A06 | a1: test. a1s, a2 and a3: the clips we embedded are split in two random halves (fixed seed 0): **eval_sel** helps choose, **eval_test** is only a test | ODC-By 1.0 |
| ASVspoof 5, train (part) | 2 of the 5 train archives (`flac_T_aa`, `flac_T_ab`): 73,000 of 182,357 clips. Fakes from 8 modern systems (A01 to A08), real speech crowdsourced from about 2,000 speakers in many rooms and on many microphones. We embedded damaged copies of 7,594 real and 12,500 fake clips | training a3 | ODC-By 1.0; real speech CC BY 4.0 (from MLS) |
| ASVspoof 5, dev (part) | 1 of the 3 dev archives (`flac_D_aa`): 47,400 of 140,950 clips. 8 other systems (A09 to A16), other speakers. We embedded 3,000 real and 3,000 fake, and (for a3) a damaged copy of the same 6,000 | choosing a2 and a3 | ODC-By 1.0; real speech CC BY 4.0 |
| CodecFake | 10,005 fake clips: VCTK sentences re-made by 15 neural audio codecs (the building block of today's voice-cloning systems), 667 per codec | training a2 and a3 | CC BY 4.0 (built on VCTK, CC BY 4.0) |
| In-the-Wild | 31,779 clips of 58 celebrities and politicians from online videos (19,963 real, 11,816 fake) | test only, never used for any choice | CC BY-NC 4.0 |

ASVspoof 2019 LA is built from clean studio recordings (the VCTK corpus) of read English sentences. In-the-Wild is the opposite: real-world audio with background noise, music, compression and many recording setups. A detector that only learned "studio vs synthetic" will struggle on it, which is exactly why it is the held-out test.

Where the files live (the data folder is not in git):

```
data/raw/audio/_archives/LA.zip                       # 7,640,952,520 bytes, MD5 30c98f11d8b2bc21f2c257bfd78bb5c5
data/raw/audio/_archives/release_in_the_wild.zip      # 8,161,489,023 bytes, SHA-256 46665f30a6758b45642c659c7e0e80da2ba3d121b064716824a56a010ddf9e1a
data/raw/audio/asvspoof2019_la/LA/...                 # LA.zip extracted
data/raw/audio/in_the_wild/release_in_the_wild/...    # meta.csv + 0.wav ... 31778.wav
data/raw/audio/_archives/asvspoof5/                   # ASVspoof5_protocols.tar.gz, flac_T_aa.tar, flac_T_ab.tar, flac_D_aa.tar (deleted once extracted and checked)
data/raw/audio/asvspoof5/                             # protocol .tsv files + flac_T/ and flac_D/ (the archives extracted)
data/raw/audio/codecfake/data/*.parquet               # 16 of the 161 shards (one from the middle of each codec's block)
data/raw/audio/codecfake/{clips.csv,wav/}             # the 10,005 clips we use, written by `python -m ml.audio.datasets codecfake`
```

ASVspoof 5 MD5 checksums (from its README.txt; our downloads match): `flac_T_aa.tar` b0cc86b14826a7701b52aad4f53daf9c, `flac_T_ab.tar` d05be3f4be7a343fdbdd0ed29fdff2e1, `flac_D_aa.tar` df0be44957623991028cce59792beb17, `ASVspoof5_protocols.tar.gz` 865d0e894ea9f686f0f37e5ae3ae3616.

Downloads:

- ASVspoof 2019: https://datashare.ed.ac.uk/handle/10283/3336 (only `LA.zip`; the MD5 above is the one the repository publishes).
- In-the-Wild: https://deepfake-total.com/in_the_wild, which links to the file on Hugging Face (`mueller91/In-The-Wild`, `release_in_the_wild.zip`; the SHA-256 above is the one Hugging Face publishes).
- ASVspoof 5: https://zenodo.org/records/14498691 (open, no sign-up). Zenodo was slow (about 0.5 MB/s per connection), so the tar files came from the organisers' Hugging Face copy, `jungjee/asvspoof5` (not gated, same files: the MD5 checksums above match the Zenodo README). Only the protocol file and the three archives above were downloaded (about 21.7 GB).
- CodecFake: https://huggingface.co/datasets/rogertseng/CodecFake (not gated). The parquet shards are grouped by codec (about 10 shards per codec), so one shard from the middle of each block (`train-00005`, `-00015`, ..., `-00155`) covers all 16 groups; one group is real VCTK speech and is skipped. About 9.6 GB.

## 2. Licences and required attribution

**ASVspoof 2019 (ODC-By 1.0).** The database may be used, also commercially, if it is credited. Any public product or page that uses the model trained on it must show:

> Contains information from the ASVspoof 2019 database, which is made available under the ODC Attribution License.
> Yamagishi, J., Todisco, M., Sahidullah, M., Delgado, H., Wang, X., Evans, N., Kinnunen, T., Lee, K. A., Vestman, V., Nautsch, A. (2019). ASVspoof 2019: The 3rd Automatic Speaker Verification Spoofing and Countermeasures Challenge database. University of Edinburgh, The Centre for Speech Technology Research (CSTR). https://doi.org/10.7488/ds/2555

The real speech inside it comes from VCTK (CC BY 4.0), so credit VCTK too: Yamagishi, J., Veaux, C., MacDonald, K. (2019). CSTR VCTK Corpus, University of Edinburgh. https://doi.org/10.7488/ds/2645

**In-the-Wild (CC BY-NC 4.0).** Non-commercial use only, with credit. SixFingers is non-commercial, and In-the-Wild is used only to measure the detector; it is never used to train or tune it. If the project ever becomes commercial, stop using it (or ask the authors at Fraunhofer AISEC). Credit:

> In-the-Wild dataset by N. M. Müller, P. Czempin, F. Dieckmann, A. Froghyar, K. Böttinger, "Does Audio Deepfake Detection Generalize?", Interspeech 2022 (arXiv:2203.16263), licensed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/). Used unmodified, for evaluation only.

Note: the Hugging Face copy's metadata says `cc-by-sa-4.0`, but the official page says CC BY-NC 4.0. We follow the stricter official page.

**ASVspoof 5 (ODC-By 1.0; real speech CC BY 4.0).** Commercial use is allowed with credit. a3 trains on ASVspoof 5 train clips, and the layer, `C`, temperature and threshold of a2 and a3 are chosen partly on ASVspoof 5 dev clips, so any public product or page that uses a2 or a3 must show:

> Contains information from the ASVspoof 5 database, which is made available under the ODC Attribution License.
> Wang, X., Delgado, H., Tak, H., Jung, J., Shim, H., Todisco, M., Kukanov, I., Liu, X., Sahidullah, M., Kinnunen, T. H., Evans, N., Lee, K. A., Yamagishi, J. (2024). ASVspoof 5: crowdsourced speech data, deepfakes, and adversarial attacks at scale. Proc. ASVspoof 2024 Workshop, 1-8. https://doi.org/10.21437/ASVspoof.2024-1

Its real speech comes from Multilingual LibriSpeech (CC BY 4.0), so credit it too: Pratap, V., Xu, Q., Sriram, A., Synnaeve, G., Collobert, R. (2020). MLS: A Large-Scale Multilingual Dataset for Speech Research. Interspeech 2020. https://doi.org/10.21437/Interspeech.2020-2826 (https://www.openslr.org/94/, licence https://creativecommons.org/licenses/by/4.0/). The README also says any malicious use of the challenge outcomes is prohibited.

**CodecFake (CC BY 4.0).** Commercial use is allowed with credit. Credit (a2 and a3 are trained on it):

> CodecFake dataset by H. Wu, Y. Tseng, H. Lee, "CodecFake: Enhancing Anti-Spoofing Models Against Deepfake Audios from Codec-Based Speech Synthesis Systems", Interspeech 2024 (arXiv:2406.07237), https://huggingface.co/datasets/rogertseng/CodecFake, licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). We use a 10,005-clip sample, unmodified apart from the training-time degradations described below.

It is built on VCTK, so the VCTK credit above applies to it as well.

**XLS-R (Apache-2.0).** `facebook/wav2vec2-xls-r-300m` by Meta AI (Babu et al., "XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale", 2021). Apache-2.0 allows any use; keep the licence notice if the weights are redistributed.

## 3. Method

Code: `ml/audio/`.

1. **Decode** (`features.decode`). ffmpeg turns any audio file into 16 kHz mono floating-point samples. It runs with `-nostdin` and `-protocol_whitelist file,pipe`, so a crafted file can never make it fetch a URL, and with a 30 s timeout. At most the first 30 s are read.
2. **Trim silence** (`features.trim_silence`). Leading and trailing stretches quieter than 40 dB below the loudest 10 ms frame are cut. Why: in ASVspoof 2019 LA, the length of the silence at the start of a clip alone predicts real vs fake surprisingly well (Müller et al., "Speech is Silver, Silence is Golden", 2021). A probe trained on untrimmed clips partly learns "how much silence", which means nothing on real-world audio.
3. **Crop** (`features.crop`). Take the first 4 s. Shorter clips are repeated end to end until they are 4 s long, so every crop has the same length and clips can be processed in batches.
4. **Embed** (`features.XLSR`). XLS-R 300M is a speech model with 24 transformer layers, pre-trained by Meta on 436,000 hours of speech in 128 languages. It was never trained to spot fakes. Each layer gives 1024 numbers for every 20 ms of audio; we average them over time ("mean-pool") to get one 1024-number vector per clip. Different layers capture different things (lower: sound texture; higher: words and meaning), so we keep layers 3, 6, 9 and 12 and let the selection sets pick one. Layers 13 to 24 are never run: on an M2 Mac the full model embeds about 4 clips per second, and stopping at layer 12 roughly doubles that. Studies of these models for fake-speech detection mostly find the lower and middle layers the most useful, but we did not test 13 to 24 ourselves. On the Mac GPU the model runs in float16 (about 1.4x faster; its embeddings match float32 to a cosine similarity above 0.9999).
5. **Damage the training clips** (`features.augment`, a2 and a3). Real-world recordings are noisy, compressed and often recorded at a low sample rate; studio training clips are none of these. So a2 trains on the clean LA train clips plus a damaged copy of each, and on damaged copies only of the CodecFake clips. a3 adds damaged copies only of the ASVspoof 5 train clips, and its selection pool also holds a damaged copy of each ASVspoof 5 dev clip. Each of four steps happens with probability 1/2: a loudness change (-20 to +10 dB, clipped at full scale like an overdriven microphone), background noise (white, pink or brown noise at 10 to 30 dB signal-to-noise ratio), a lower sample rate (8, 11.025, 12 or 14 kHz and back to 16 kHz, like a phone line), and a lossy codec round trip (MP3 or AAC at 24 to 64 kbit/s, Opus at 12 to 32 kbit/s). The damage is random but fixed: the same file always gets the same damage (seeded by the file name and `AUG_SEED`). ffmpeg does the codec step with `-nostdin` and only reads and writes local temporary files. About 1 in 8 copies gets no noise, resampling or codec, so it stays close to clean (a loudness change alone does little, because each clip is normalized before embedding). Test sets are never damaged. Sets with damaged copies have names ending in `_aug`.
6. **Cache** (`ml/audio/extract.py`). Embedding is the slow part, so results are saved as float16 in `data/features/audio/<set>.npz` (one file per set, holding every layer).
7. **Probe** (`ml/audio/train.py`). For each layer: standardize the numbers and fit a logistic regression on the training sets (class-balanced, because they hold more fakes than real clips), for each `C` in {0.001, 0.01, 0.1, 1}. Smaller `C` means a simpler model (stronger regularization).
8. **Choose and calibrate.** Two rules exist:
   - **Rule `dev` (a1).** Calibrate with temperature scaling on LA dev; per layer, the `C` with the lowest calibrated dev log-loss wins; the layer with the highest dev AUROC wins overall. Threshold: 5% of the real LA dev clips flagged.
   - **Rule `sel` (a1s, a2, a3, the default).** LA dev turned out to be too easy to choose with (every layer scored above 0.9998 AUROC), so a harder selection pool is used. The embedded LA eval clips are split into two random halves with a fixed seed (`SPLIT_SEED = 0`, same share of fakes in each). One half, **eval_sel**, has 11 attack systems the training set never saw, so it separates good layers from bad ones; it joins LA dev (and, for a2 and a3, the ASVspoof 5 dev clips; for a3, also their damaged copies) in the selection pool. The `(layer, C)` pair with the highest AUROC on the pooled selection clips wins. Temperature scaling is fitted on the same pool, and the threshold is set so that 5% of the pooled real selection clips are flagged.
   The other half, **eval_test**, and In-the-Wild are only scored at the end; no choice looks at them.

How many clips were embedded: at most 5,000 real and 5,000 fake clips per set, picked at random with a fixed seed (`--per-class 5000`). LA train and LA dev have fewer than 5,000 real clips, so all of their real clips are used. For a2: 10,000 of the 10,005 CodecFake clips, and 3,000 real plus 3,000 fake clips from the downloaded ASVspoof 5 dev archive (same random-pick rule). For a3: all 7,594 real and 12,500 of the fake clips from the two downloaded ASVspoof 5 train archives (damaged copies only), and a damaged copy of the same 6,000 ASVspoof 5 dev clips.

Run it (from the repo root, with `SIXFINGERS_DATA_DIR` / `SIXFINGERS_MODELS_DIR` pointing at the data and model folders):

```
python -m ml.audio.extract --per-class 5000   # LA + ITW sets by default; caches embeddings (GPU, about 2.5 h on an M2)
python -m ml.audio.train --version a1 --rule dev   # a1: fits, picks on LA dev, reports (CPU only)
python -m ml.audio.train --version a1s             # a1s: same data, rule sel
```

For a2 (about 1.2 more hours of embedding on an M2; training takes about a minute):

```
python -m ml.audio.datasets codecfake --n 10000                    # unpack the CodecFake sample
python -m ml.audio.extract --sets la_train_aug --per-class 5000    # damaged copies of the LA train clips
python -m ml.audio.extract --sets codecfake_aug --per-class 10000
python -m ml.audio.extract --sets asv5_dev --per-class 3000
python -m ml.audio.train --version a2 --train la_train la_train_aug codecfake_aug --sel la_dev asv5_dev
```

For a3 (about 70 more minutes of embedding on an M2: 54 min for the train clips, 15 min for the dev copies; training takes a few minutes):

```
python -m ml.audio.extract --sets asv5_train_aug --per-class 12500   # damaged copies: 7,594 real + 12,500 fake
python -m ml.audio.extract --sets asv5_dev_aug --per-class 3000      # damaged copies of the asv5_dev clips
python -m ml.audio.train --version a3 --train la_train la_train_aug asv5_train_aug codecfake_aug \
    --sel la_dev asv5_dev asv5_dev_aug
```

The model folder holds `probe.json` (same format as the image probes) and `meta.json` (layer, threshold, input settings, clip counts and every metric below).

## 4. How to read the numbers

Label 1 = fake. "Positive" = flagged as fake.

- **EER** (equal error rate): move the threshold until the share of real clips wrongly flagged equals the share of fakes missed; EER is that share. Lower is better; 0.5 is a coin flip. It is the standard number in anti-spoofing papers.
- **AUROC**: the chance that a random fake scores above a random real clip (1.0 perfect, 0.5 coin flip).
- **TPR@5%FPR**: the share of fakes caught when the threshold lets only 5% of real clips be wrongly flagged.
- **Accuracy at the shipped threshold**: the share of all clips labelled correctly when the threshold chosen on the selection sets is used as is. Our test sets are half real, half fake, so 0.5 is a coin flip.
- **At the shipped threshold**: fake recall (share of fakes caught) and real FPR (share of real clips wrongly flagged), using that same threshold. On a new kind of data this threshold can be far from the right one, and these two numbers show how far.

## 5. Results

Summary on the two test sets (no choice looked at either). One caveat: before rule `sel` was designed, every a1 layer was scored on LA eval and In-the-Wild as a check (see the a1 section). Rule `sel` then picked layer 6, the layer that check had liked. The rule never sees the test sets, but its design came after that look, so treat the a1s, a2 and a3 test numbers as slightly optimistic. eval_test is the half of the embedded LA eval clips that was not used for choosing; a1 never used LA eval at all, and is scored on the same half here so the rows compare like with like.

| Model | Test set | EER | AUROC | TPR@5%FPR | Real FPR at threshold |
|---|---|---|---|---|---|
| a1 (rule dev) | eval_test | 0.097 | 0.970 | 0.854 | 0.069 |
| a1s (rule sel) | eval_test | 0.026 | 0.997 | 0.985 | 0.032 |
| a2 (rule sel, more data, damaged copies) | eval_test | 0.037 | 0.993 | 0.971 | 0.018 |
| a3 (a2 + damaged ASVspoof 5 clips) | eval_test | 0.046 | 0.991 | 0.956 | 0.067 |
| a1 (rule dev) | In-the-Wild | 0.370 | 0.689 | 0.151 | 0.992 |
| a1s (rule sel) | In-the-Wild | 0.092 | 0.966 | 0.824 | 0.561 |
| a2 (rule sel, more data, damaged copies) | In-the-Wild | 0.103 | 0.961 | 0.819 | 0.129 |
| a3 (a2 + damaged ASVspoof 5 clips) | In-the-Wild | **0.063** | **0.979** | **0.922** | **0.033** |

In short: the honest selection rule fixes ranking (a1 to a1s), the damaged training copies plus codec fakes fix most of the threshold drift (a1s to a2), and modern fakes with crowdsourced real speech improve both on real-world audio (a2 to a3). a3 is the model to use; it gives up a little on clean studio audio (eval_test) for that.

### a1: every choice on LA dev

Model `models/audio/a1/` (XLS-R layer 3, `C` = 0.1, threshold P(fake) >= 0.0010).

| Set | Clips (real / fake) | EER | AUROC | TPR@5%FPR | Accuracy at threshold | Fake recall at threshold | Real FPR at threshold |
|---|---|---|---|---|---|---|---|
| LA dev (used for every choice) | 2,548 / 5,000 | 0.001 | 1.000 | 1.000 | 0.983 | 1.000 | 0.050 |
| LA eval (unseen attacks) | 5,000 / 5,000 | 0.098 | 0.970 | 0.857 | 0.904 | 0.876 | 0.068 |
| In-the-Wild (fully held out) | 5,000 / 5,000 | 0.370 | 0.689 | 0.151 | 0.504 | 1.000 | 0.992 |

Every layer on LA dev:

| Layer | Best `C` | Dev EER | Dev AUROC |
|---|---|---|---|
| 3 | 0.1 | 0.0012 | 0.99999 |
| 6 | 1 | 0.0020 | 0.99996 |
| 9 | 1 | 0.0040 | 0.99993 |
| 12 | 1 | 0.0044 | 0.99986 |

What this means:

- **Known attacks are easy.** On LA dev (the same 6 fake-speech systems as training) the probe is almost perfect, for every layer.
- **New attacks are harder.** On LA eval, where 11 of the 13 systems were never seen, about 1 in 10 clips lands on the wrong side at the EER point.
- **Real-world audio is much harder.** On In-the-Wild the probe is only a little better than a coin flip (AUROC 0.69).
- **The threshold does not transfer.** It is tuned on studio recordings, where real clips score almost exactly 0. Real-world recordings score much higher, so at the dev threshold 99% of real In-the-Wild clips are flagged as fake; the 0.504 accuracy is that of "call everything fake". The model is not usable on real-world audio as it stands.
- **Dev could not pick the layer.** All four layers score above 0.9998 AUROC on dev, so the differences between them are noise. Layer 3 won by 0.00003. After the choice was made, the other layers were also scored on the test sets as a check (not used for any choice): layer 6 did clearly better on both (LA eval AUROC 0.994, In-the-Wild AUROC 0.827, EER 0.249), and layers 9 and 12 sat in between. Switching to layer 6 because of this would mean choosing on the test sets, which makes their numbers look better than they really are. A fair way to choose is a selection set that is harder than LA dev (for example part of LA eval), with In-the-Wild kept as the only final test. That is what a1s and a2 below do.

### a1s: same data, harder selection set

Not shipped (a measurement of the selection rule alone): `python -m ml.audio.train --version a1s` on the a1 caches. Training data is exactly a1's (LA train). Selection pool: LA dev + eval_sel. Chosen: layer 6, `C` = 0.001, threshold P(fake) >= 0.0614.

| Set | Clips (real / fake) | EER | AUROC | TPR@5%FPR | Accuracy at threshold | Fake recall at threshold | Real FPR at threshold |
|---|---|---|---|---|---|---|---|
| LA dev (selection) | 2,548 / 5,000 | 0.004 | 1.000 | 0.999 | 0.978 | 1.000 | 0.065 |
| eval_sel (selection) | 2,500 / 2,500 | 0.028 | 0.996 | 0.984 | 0.972 | 0.978 | 0.034 |
| eval_test (test) | 2,500 / 2,500 | 0.026 | 0.997 | 0.985 | 0.973 | 0.978 | 0.032 |
| In-the-Wild (test) | 5,000 / 5,000 | 0.092 | 0.966 | 0.824 | 0.719 | 0.999 | 0.561 |

Selection AUROC for every layer and `C` (pooled LA dev + eval_sel):

| Layer | `C` = 0.001 | 0.01 | 0.1 | 1 |
|---|---|---|---|---|
| 3 | 0.9926 | 0.9912 | 0.9899 | 0.9878 |
| 6 | **0.9984** | 0.9984 | 0.9979 | 0.9971 |
| 9 | 0.9979 | 0.9980 | 0.9977 | 0.9970 |
| 12 | 0.9969 | 0.9972 | 0.9970 | 0.9959 |

What this means:

- **eval_sel separates the layers; LA dev did not.** Layer 3 is now clearly last, and layer 6 first.
- **Strong regularization matters most for real-world audio.** a1's dev rule picked `C` = 1 for layer 6 (In-the-Wild AUROC 0.827, see above); the selection pool prefers `C` = 0.001, and In-the-Wild AUROC rises to 0.966 (EER 0.092). A simpler probe leans less on studio-specific details. This is in line with published work finding that a plain logistic regression on frozen self-supervised speech features generalizes well (for example Pascu et al., Interspeech 2024).
- **eval_sel and eval_test agree** (EER 0.028 and 0.026), so the half used for choosing did not make the other half look better by luck.
- **The threshold still does not transfer.** At the chosen threshold, 56% of real In-the-Wild clips are flagged. The ranking is good; the score scale shifts on real-world audio.
- **Modern fakes, never seen in training.** As an extra check, a1s scores 0.9996 AUROC (EER 0.005) on our clean ASVspoof 5 dev clips (fakes from 8 systems built in 2023-2024); a1 scores 0.955 (EER 0.115). These clips are clean, so this says nothing about real-world conditions.

### a2: codec fakes and damaged copies

Model `models/audio/a2/` (XLS-R layer 6, `C` = 0.001, threshold P(fake) >= 0.1808).

- Training (25,160 clips): LA train (7,580 clips: 2,580 real, 5,000 fake), a damaged copy of each of them (7,580), and damaged copies of 10,000 CodecFake fakes.
- Selection pool (18,548 clips): LA dev (7,548), ASVspoof 5 dev (3,000 real, 3,000 fake, attacks A09 to A16), eval_sel (5,000).
- Tests: eval_test and In-the-Wild.

| Set | Clips (real / fake) | EER | AUROC | TPR@5%FPR | Accuracy at threshold | Fake recall at threshold | Real FPR at threshold |
|---|---|---|---|---|---|---|---|
| LA dev (selection) | 2,548 / 5,000 | 0.008 | 0.999 | 0.998 | 0.990 | 0.995 | 0.018 |
| ASVspoof 5 dev (selection) | 3,000 / 3,000 | 0.010 | 0.999 | 0.999 | 0.945 | 0.999 | 0.109 |
| eval_sel (selection) | 2,500 / 2,500 | 0.042 | 0.991 | 0.960 | 0.960 | 0.931 | 0.012 |
| eval_test (test) | 2,500 / 2,500 | 0.037 | 0.993 | 0.971 | 0.957 | 0.931 | 0.018 |
| In-the-Wild (test) | 5,000 / 5,000 | 0.103 | 0.961 | 0.819 | 0.895 | 0.919 | 0.129 |

Selection AUROC for every layer and `C` (pooled LA dev + ASVspoof 5 dev + eval_sel):

| Layer | `C` = 0.001 | 0.01 | 0.1 | 1 |
|---|---|---|---|---|
| 3 | 0.9658 | 0.9562 | 0.9495 | 0.9424 |
| 6 | **0.9956** | 0.9939 | 0.9907 | 0.9853 |
| 9 | 0.9925 | 0.9921 | 0.9898 | 0.9890 |
| 12 | 0.9867 | 0.9869 | 0.9853 | 0.9820 |

What this means:

- **The threshold now mostly transfers.** At the threshold chosen on the selection pool, 12.9% of real In-the-Wild clips are flagged (a1s: 56%, a1: 99%), and 91.9% of fakes are caught. Accuracy at the threshold is 0.895 (a1s: 0.719). This is the main gain: the damaged copies teach the probe that noise, low sample rates and compression are not signs of a fake.
- **Ranking is about the same, slightly worse.** In-the-Wild AUROC 0.961 vs 0.966 for a1s, and EER 0.103 vs 0.092; on eval_test EER 0.037 vs 0.026. With roughly 5,000 clips per class, In-the-Wild numbers carry about ±1 percentage point of sampling noise, so the AUROC change is within noise; the EER changes are small but probably real. Learning "damaged audio can be real" costs a little sharpness on clean studio audio.
- **12.9% is still above the 5% the threshold aims for.** Real-world real speech still scores higher than real speech in the selection pool. a3 below fixes this.

### a3: modern fakes and crowdsourced real speech

Model `models/audio/a3/` (XLS-R layer 6, `C` = 0.001, threshold P(fake) >= 0.0788). Commands: see section 3.

What changed from a2:

- Training (45,254 clips): a2's training data plus damaged copies of 20,094 ASVspoof 5 train clips (7,594 real, 12,500 fake from systems A01 to A08). For the first time the probe sees fakes from 2023-2024 systems, and real speech that was not recorded in a studio (crowdsourced, many rooms and microphones). Only damaged copies are used, so every one of these clips is noisy, resampled or compressed in some way (about 1 in 8 only lightly).
- Selection pool (24,548 clips): a2's pool plus a damaged copy of each of the 6,000 ASVspoof 5 dev clips (`asv5_dev_aug`).
- Tests: unchanged (eval_test and In-the-Wild). None of the new clips is in a test set: ASVspoof 5 is a separate database, and its train and dev clips share no file.

Choosing on damaged dev copies was deliberate. The threshold is set so that 5% of the real clips in the selection pool are flagged. If that pool holds only clean audio, the threshold ends up too low for noisy real recordings, which score higher (a2 flagged 12.9% of real In-the-Wild clips instead of 5%). Adding damaged real clips moves the threshold to where messy real-world audio sits. These clips are copies of selection clips, never of test clips, so the test sets stay untouched.

| Set | Clips (real / fake) | EER | AUROC | TPR@5%FPR | Accuracy at threshold | Fake recall at threshold | Real FPR at threshold |
|---|---|---|---|---|---|---|---|
| LA dev (selection) | 2,548 / 5,000 | 0.011 | 0.999 | 0.996 | 0.977 | 0.997 | 0.064 |
| ASVspoof 5 dev (selection) | 3,000 / 3,000 | 0.005 | 1.000 | 0.999 | 0.992 | 0.998 | 0.014 |
| ASVspoof 5 dev, damaged (selection) | 3,000 / 3,000 | 0.010 | 0.999 | 0.999 | 0.972 | 0.999 | 0.055 |
| eval_sel (selection) | 2,500 / 2,500 | 0.050 | 0.989 | 0.950 | 0.942 | 0.958 | 0.073 |
| eval_test (test) | 2,500 / 2,500 | 0.046 | 0.991 | 0.956 | 0.949 | 0.965 | 0.067 |
| In-the-Wild (test) | 5,000 / 5,000 | 0.063 | 0.979 | 0.922 | 0.913 | 0.859 | 0.033 |

Selection AUROC for every layer and `C` (pooled LA dev + ASVspoof 5 dev + its damaged copy + eval_sel):

| Layer | `C` = 0.001 | 0.01 | 0.1 | 1 |
|---|---|---|---|---|
| 3 | 0.9876 | 0.9911 | 0.9915 | 0.9902 |
| 6 | **0.9980** | 0.9977 | 0.9968 | 0.9956 |
| 9 | 0.9972 | 0.9974 | 0.9971 | 0.9965 |
| 12 | 0.9955 | 0.9958 | 0.9950 | 0.9937 |

What this means:

- **Better on real-world audio, in ranking and in threshold.** On In-the-Wild, EER falls from 0.103 to 0.063, AUROC rises from 0.961 to 0.979, and at 5% false alarms 92.2% of fakes are caught (a2: 81.9%). At the shipped threshold, 3.3% of real clips are flagged (a2: 12.9%), now below the 5% target. To check these are not luck, we resampled the In-the-Wild clips 500 times (bootstrap, paired: both models scored on the same resampled clips). The 95% range of the change is -0.045 to -0.035 for EER, +0.016 to +0.020 for AUROC, and -10.5 to -8.7 percentage points for the real FPR at the threshold. All clearly real gains.
- **Fewer fakes caught at the threshold on In-the-Wild.** Fake recall at the threshold drops from 91.9% to 85.9%: a threshold that flags fewer real clips also lets more fakes through. (a3's threshold, 0.079, is lower than a2's 0.181 as a number, but each model has its own score scale, so the two numbers cannot be compared directly.) Overall accuracy at the threshold still rises from 0.895 to 0.913. Fewer false alarms on real voices for a few more missed fakes is the trade we want for a check people see.
- **A little worse on clean studio audio.** On eval_test, EER rises from 0.037 to 0.046 and the real FPR at the threshold from 1.8% to 6.7% (bootstrap ranges: EER +0.004 to +0.014, real FPR +4.1 to +5.8 points). The threshold now suits noisy audio, so clean studio real speech is flagged a bit more often than the 5% target.
- **Same layer and `C` as a1s and a2** (layer 6, `C` = 0.001). The new data changed the probe's weights and the threshold, not the choice.

## 6. Limits

- **Few modern fakes in training.** a3's training fakes are 2018-era systems (ASVspoof 2019), speech re-made by neural audio codecs (CodecFake), and 8 research systems from 2023-2024 (ASVspoof 5 train, A01 to A08). Modern commercial voice cloning is not covered by any set here. In-the-Wild's fakes are from before 2022. Expect worse results on current voices than any number above.
- **Read English speech in training.** a3's real training clips are all read English: clean studio recordings (VCTK, through ASVspoof 2019) with damaged copies of them, and damaged copies of crowdsourced ASVspoof 5 recordings. Conversation, music, crowds, other languages and real phone lines were never seen. The damage is synthetic (generated noise, codec round trips), not recorded.
- **Only the first 4 s.** Long clips are judged on their first 4 s after silence trimming. A fake that starts with real speech would be missed.
- **Threshold drift.** The threshold is set on the selection pool. On other audio the scores shift. a3 flags 3.3% of real In-the-Wild clips (a2: 12.9%, a1s: 56%, a1: 99%) but 6.7% of real clean studio clips (eval_test), instead of the intended 5%. On audio unlike either set, the false-alarm rate is unknown.
- **CodecFake shares speakers with LA.** CodecFake re-makes VCTK sentences, and LA's real clips (including eval_test's) are VCTK speakers too. A probe could learn "this VCTK voice re-made by a codec is fake", which is harmless for the test numbers (it can only push real VCTK clips towards fake) but means CodecFake adds no new voices.
- **Probabilities follow the selection mix.** Temperature scaling is fitted on the selection pool, which is about 55% fake for a3 (57% for a2), so the probability assumes that mix. It is not the chance that a random recording from the internet is fake.
- **No speaker check.** It asks "is this synthetic?", not "is this the person it claims to be?". Real speech by an impersonator counts as real.
- **Subsampled test sets.** See the counts in the results tables; eval_test has 2,500 clips per class, so its numbers carry sampling noise of roughly ±1 to 2 percentage points; In-the-Wild about ±1.

## 7. In the API

The API can run this model, as a prototype that is off by default. Start it with `SIXFINGERS_ENABLE_AUDIO=1` and it gains `POST /v1/check-audio`, and video results get an audio score for their soundtrack. The website then shows an Audio tab. Details are in [api.md](api.md) under **Audio checks**.

- The model comes from `models/audio/a3/` (`probe.json` and `meta.json`); `SIXFINGERS_AUDIO_MODEL` picks another folder (for example `a2`).
- The input is prepared the same way as in training (`ml.audio.features`: 16 kHz mono, silence trimmed at -40 dB, first 4 s), so the scores mean the same thing as in the tables above.
- Before the score is used, a speech gate checks that those 4 s sound like speech (section 8). If they do not, there is no band, only the note "No clear speech found, so the voice was not checked." (`POST /v1/check-audio` answers `422` with that text; a video result gets `audio: {analysed: false, note: ...}`).
- The answer is `p_fake` and a band: `likely_ai` at or above the threshold in `meta.json` (0.079 for a3), `likely_real` below it.
- Every limit in section 6 applies. In short: only the first 4 seconds count, and the model learned from read English speech. Clips with no clear speech get no voice verdict (section 8).
- Claude cannot hear audio, so there is no written explanation, only the score.

## 8. Speech gate: only check a voice when there is speech

**The problem.** The probe was trained on speech only. Give it music or street noise and it still answers, and the answer means nothing. We measured it (below): without a gate, **54.9%** of 390 real music and ambient clips were called `likely_ai`. In a video, one such flag makes the whole video `likely_ai` (see [video.md](video.md)), so a real video with only a music track could be flagged.

**The fix.** A second small probe, the gate, reads the same XLS-R embedding (layer 6, the same 4 s) that the voice probe reads, so it costs no extra model run. It answers "is this speech?". Below its threshold, the API returns no voice band and the video's overall verdict uses the picture only. The gate lives in `models/audio/<model>/gate.json`; if that file is missing the API logs a warning and scores everything, as before.

**How its settings were chosen** (all on data that is not used for the numbers below):

| | Clips | Where from |
|---|---|---|
| Speech examples | 5,500 (70% fitted, 30% held out) | the training sets only: `la_train`, `asv5_train_aug`, `codecfake_aug` (real and fake speech both count as speech) |
| Non-speech examples | 1,600 (70% fitted, 30% held out) | FSD50K **dev** clips (CC BY 4.0 as a set; each clip has its own Creative Commons licence): 800 music, 800 other sounds. Clips labelled speech, singing, laughter or other human voice sounds were left out. |

It is a logistic probe (`C` = 0.1). The threshold rule: let at most 1% of the held-out non-speech clips through. On those held-out tuning clips that let through 1.0% of non-speech and gated out 0.06% of speech. Honest history: the first rule tried was "gate out 1% of held-out speech". After seeing the validation and In-the-Wild gating rates for it (20% of ASVspoof 5 dev fakes and 24% of real In-the-Wild clips gated out, far too strict) we switched to the rule above. That is one revision made while looking at those validation numbers, so they are slightly optimistic; the rule was fixed before the gate was ever run on the non-speech test clips.

**Measured with the gate on.** Non-speech test set (390 clips, never used for tuning, scored by the live `AudioScorer`, a3): FSD50K **eval** clips (100 music, 40 music with singing, 100 other sounds) and 150 ESC-50 clips (30 from each of 5 groups; ESC-50 is CC BY-NC 3.0, fine for this non-commercial project). The `likely_ai` column counts clips that still get a voice verdict of `likely_ai`.

| Group | Clips | `likely_ai` without gate | Still get any voice verdict | Still `likely_ai` |
|---|---|---|---|---|
| FSD50K music | 100 | 39% | 4% | 4% |
| FSD50K music with singing | 40 | 25% | 2.5% | 0% |
| FSD50K other sounds | 100 | 68% | 1% | 1% |
| ESC-50 animals | 30 | 63% | 3.3% | 0% |
| ESC-50 nature, water, wind | 30 | 63% | 3.3% | 3.3% |
| ESC-50 human non-speech (cough, laugh, cry...) | 30 | 80% | 10% | 10% |
| ESC-50 indoor sounds | 30 | 77% | 0% | 0% |
| ESC-50 city sounds | 30 | 40% | 0% | 0% |
| **All 390** | 390 | **54.9%** (214 clips) | **2.8%** (11 clips) | **2.3%** (9 clips) |

With 390 clips, a share has an uncertainty of roughly 1 to 2 points. Human sounds that are not speech (coughs, laughs, a crying baby) are the weak spot: 3 of 30 still got a verdict.

**Real and fake speech wrongly gated out** (so they get no voice verdict). Computed from the cached embeddings of the existing audio validation sets, which are the same crops the live scorer uses (stored as float16):

| Set | Real speech | Fake speech |
|---|---|---|
| LA dev | 1 of 2,548 (0.04%) | 0 of 5,000 (0%) |
| ASVspoof 5 dev | 0 of 3,000 (0%) | 53 of 3,000 (1.8%) |
| ASVspoof 5 dev, damaged copies | 0 of 3,000 (0%) | 39 of 3,000 (1.3%) |
| In-the-Wild (report only, never used for any choice) | 98 of 5,000 (2.0%) | 30 of 5,000 (0.6%) |

Gated-out fakes are fakes that go unchecked, not wrongly flagged real clips. The validation sets also helped choose the voice probe (a3), not the gate.

**Speech with music underneath.** We mixed 60 real and 60 fake clean LA dev clips with FSD50K eval music: at 10 dB (speech clearly louder) 18% of the real and 12% of the fake mixes were gated out; at 0 dB (music as loud as the voice) 50% and 30%. So a voice over loud music often gets no voice check. That is the safe direction (no false flag), but it means a fake voice over loud music is missed.

**Limits of the gate.**
- It judges the first 4 s after silence trimming, like the voice probe. A video that opens with a music intro and speaks later gets no voice check.
- The speech examples are studio and crowdsourced read English plus damaged copies. The gate has not seen conversations or other languages as speech examples, and the real-world check is In-the-Wild only (2.0% of real speech gated out there).
- Singing is mostly treated as not speech (39 of 40 singing clips gated out).
- The test set is small (390 clips, 11 let through); do not read the percentages to the decimal.
