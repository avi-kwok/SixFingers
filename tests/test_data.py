import io

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageFilter

from ml.data import audit, generate_paid, normalize, splits
from ml.data.augment import augment


def _img(w=300, h=200, seed=0):
    return Image.fromarray(np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8))


def _enc(img, fmt, **kw):
    b = io.BytesIO()
    img.save(b, fmt, **kw)
    return b.getvalue()


def test_normalizer_same_output_format_for_png_and_jpeg_and_strips_metadata():
    img = _img()
    exif = Image.Exif()
    exif[0x0131] = "SomeSoftware"
    outs = [normalize.normalize_bytes(_enc(img, "PNG")),
            normalize.normalize_bytes(_enc(img, "JPEG", quality=60, exif=exif))]
    for o in outs:
        im = Image.open(io.BytesIO(o))
        assert im.format == "JPEG" and im.size == (512, 512)
        assert not im.getexif() and "icc_profile" not in im.info
    assert Image.open(io.BytesIO(outs[0])).quantization == Image.open(io.BytesIO(outs[1])).quantization
    assert normalize.normalize_bytes(_enc(img, "PNG")) == outs[0]  # deterministic


def test_splits_holdout_and_groups():
    df = pd.DataFrame({"generator": ["flux"] * 5 + ["real_raise"] * 5 + ["a", "b"] * 100,
                       "group": [f"g{i}" for i in range(10)] + [f"p{i // 2}" for i in range(200)]})
    s = splits.assign_splits(df, "flux")
    assert set(s[df.generator.isin(["flux", "real_raise"])]) == {"report_test"}
    assert "report_test" not in set(s[~df.generator.isin(["flux", "real_raise"])])
    rest = df[s != "report_test"].assign(split=s)
    assert (rest.groupby("group").split.nunique() == 1).all()  # a pair never straddles splits
    assert set(rest.split) == {"train", "val", "gate"}


def test_freeze_writes_once(tmp_path, monkeypatch):
    monkeypatch.setattr(splits, "FROZEN_DIR", tmp_path)
    assert splits.read_frozen() is None
    assert splits.freeze(["b", "a"]) == (splits.digest(["a", "b"]), True)
    assert splits.read_frozen() == {"a", "b"}
    before = splits.frozen_path().read_bytes()
    assert splits.freeze(["a", "b", "c"])[1] is False
    assert splits.frozen_path().read_bytes() == before
    assert splits.read_frozen() == {"a", "b"}


def _audit_manifest(tmp_path, make):
    # make(i) -> (label, PIL image, extra JPEG save kwargs); all files saved as JPEG quality 90
    rows = []
    for i in range(60):
        label, img, kw = make(i)
        p = tmp_path / f"{i}.jpg"
        img.save(p, "JPEG", quality=90, **kw)
        rows.append({"path": str(p), "label": label})
    return pd.DataFrame(rows)


def test_audit_ignores_file_size_texture(tmp_path):
    def make(i):
        if i % 2:  # label 1: smooth, flat or blurred
            img = (Image.new("RGB", (64, 64), (i * 3 % 256, 90, 200)) if i % 4 == 1
                   else _img(64, 64, i).filter(ImageFilter.GaussianBlur(6)))
            return 1, img, {}
        return 0, _img(64, 64, i), {}  # label 0: noisy
    r = audit.audit(_audit_manifest(tmp_path, make))
    assert r["passed"]
    assert abs(r["file_size_auroc"] - 0.5) > 0.3


def test_audit_fails_dimension_leak(tmp_path):
    def make(i):
        return (1, _img(64, 48, i), {}) if i % 2 else (0, _img(64, 64, i), {})
    assert not audit.audit(_audit_manifest(tmp_path, make))["passed"]


def test_audit_fails_metadata_leak(tmp_path):
    def make(i):
        ex = Image.Exif()
        ex[0x010F] = "Cam"
        return (0, _img(64, 64, i), {"exif": ex}) if i % 2 == 0 else (1, _img(64, 64, i), {})
    assert not audit.audit(_audit_manifest(tmp_path, make))["passed"]


def test_augment_deterministic_and_varied():
    img = _img(400, 400)
    a, b, c = augment(img, 1), augment(img, 1), augment(img, 2)
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert a.size != c.size or not np.array_equal(np.asarray(a), np.asarray(c))


def test_paid_generation_requires_approval(monkeypatch):
    monkeypatch.setattr("sys.argv", ["x"])
    with pytest.raises(SystemExit) as e:
        generate_paid.main()
    assert e.value.code == 2  # argparse: --approve-spend missing
    monkeypatch.setattr("sys.argv", ["x", "--approve-spend", "0"])
    with pytest.raises(SystemExit, match="refusing"):
        generate_paid.main()
