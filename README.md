# SixFingers

Upload an image, find out whether it looks AI-generated. Fully AI-generated stills only (JPEG/PNG/WebP).

See `docs/ARCHITECTURE.md` for how it fits together.

Two checks look at every image:

- **Detector.** Frozen CLIP, DINOv2 and SigLIP backbones with small trained probes on top.
- **Visual judge.** A vision-language model, **Claude Haiku 5.5** (`claude-haiku-5-5`), looks at the image alone (it is not shown the detector scores). It gives a verdict and boxes the regions that gave it away. Its confidence is calibrated (Platt scaling), so 80% means right about 80% of the time. On the held-out test set it is right on 96% of images (`/scorecard` has the details). If the judge is unavailable, the detector's answer is shown alone.

The live model is v16. On the held-out test set it catches 96% of AI images from a generator it never trained on. It still wrongly flags about 7% of real camera photos (10% when the photos are recompressed). Full numbers: `docs/results.md`.

    make setup && make test

## Licence

The code is under the MIT licence (see `LICENSE`). The MIT licence covers the code only. The datasets and the model weights trained on them keep their own licences, and some of those allow non-commercial use only. See `docs/data.md` for each source.
