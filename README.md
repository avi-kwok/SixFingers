# SixFingers

Upload an image, find out whether it looks AI-generated. Fully AI-generated stills only (JPEG/PNG/WebP).

See `docs/ARCHITECTURE.md` for how it fits together.

The live model is v15. On the held-out test set it catches 96% of AI images from a generator it never trained on. It still wrongly flags about 18% of real camera photos. Full numbers: `docs/results.md`.

    make setup && make test

## Licence

The code is under the MIT licence (see `LICENSE`). The MIT licence covers the code only. The datasets and the model weights trained on them keep their own licences, and some of those allow non-commercial use only. See `docs/data.md` for each source.
