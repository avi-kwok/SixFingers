"""Free local generation of content-matched fakes: one image per COCO caption already in data/raw/coco.
Same caption + same seed -> same image, so reruns are resumable and reproducible."""
import argparse
import zlib

from ml.data.sources import append_index, load_registry, read_index, source_dir

MODELS = {
    "sdxl_turbo": dict(repo="stabilityai/sdxl-turbo", source="sdxl_turbo", pipe="AutoPipelineForText2Image",
                       kw=dict(num_inference_steps=1, guidance_scale=0.0, height=512, width=512)),
    # not exercised by tests or the smoke run (~24 GB of weights)
    "flux_schnell": dict(repo="black-forest-labs/FLUX.1-schnell", source="flux_schnell_local", pipe="FluxPipeline",
                         kw=dict(num_inference_steps=4, guidance_scale=0.0, height=512, width=512)),
}


def load_pipe(cfg: dict):
    import diffusers
    import torch

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if dev == "mps" else torch.float32
    kwargs = {"variant": "fp16"} if cfg["pipe"] == "AutoPipelineForText2Image" else {}
    pipe = getattr(diffusers, cfg["pipe"]).from_pretrained(cfg["repo"], torch_dtype=dtype, **kwargs)
    return pipe.to(dev), dev


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--limit", type=int, default=200)
    grp.add_argument("--grow", type=int, help="generate N more images than already indexed")
    ap.add_argument("--model", choices=list(MODELS), default="sdxl_turbo")
    a = ap.parse_args()
    cfg = MODELS[a.model]
    src = next(s for s in load_registry()["sources"] if s["name"] == cfg["source"])
    limit = a.limit if a.grow is None else len(read_index(src["name"])) + a.grow
    prompts = read_index("coco")[:limit]
    if not prompts:
        raise SystemExit("no COCO index; run: python -m ml.data.download --source coco first")
    done = {r["file"] for r in read_index(src["name"])}
    todo = [p for p in prompts if p["file"].replace(".jpg", ".png") not in done]
    if not todo:
        print("nothing to do")
        return
    import torch
    import numpy as np

    pipe, dev = load_pipe(cfg)
    root = source_dir(src["name"])
    root.mkdir(parents=True, exist_ok=True)
    for p in todo:
        out = p["file"].replace(".jpg", ".png")
        gen = torch.Generator("cpu").manual_seed(zlib.crc32(p["file"].encode()))
        img = pipe(p["caption"], generator=gen, **cfg["kw"]).images[0]
        if np.asarray(img).max() < 5:  # fp16 VAE on MPS can return all-black; refuse to keep it
            raise SystemExit(f"black output for {p['file']} (fp16 VAE issue on {dev}); not saved")
        img.save(root / out)
        append_index(src["name"], {"file": out, "source": src["name"], "generator": src["generator"],
                                   "side": "ai", "license": src["license"], "group": p["group"],
                                   "caption": p["caption"]})
        print("generated", out, flush=True)


if __name__ == "__main__":
    main()
