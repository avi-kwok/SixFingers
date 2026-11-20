import numpy as np
from PIL import Image

from api.detector import DetectorStore
from tests.api.conftest import StubEmbedder, make_model


class BatchStub(StubEmbedder):
    def embed_many(self, backbone, images):
        return np.vstack([self.embed(backbone, im) for im in images])


def test_batched_scores_match_single(tmp_path):
    make_model(tmp_path, "v1")
    (tmp_path / "current").symlink_to("v1")
    ims = [Image.new("RGB", (64, 64), (v, v, v)) for v in (0, 60, 128, 200, 255)]
    for emb in (StubEmbedder(), BatchStub()):
        store = DetectorStore(emb, tmp_path)
        batch, _ = store.score_many(ims)
        for im, b in zip(ims, batch):
            s, _ = store.score(im)
            assert abs(s.score - b.score) < 1e-5 and s.band == b.band
            assert all(abs(s.backbones[k] - b.backbones[k]) < 1e-5 for k in s.backbones)
