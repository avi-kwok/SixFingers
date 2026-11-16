"""Split assignment + frozen report_test hash list.
report_test = every image of the held-out generator + the RAISE reals. Everything else is split by a hash
of its content group (so a caption's real/fake pair never straddles splits): 70% train, 15% val, 15% gate."""
import hashlib
from pathlib import Path

import pandas as pd

FROZEN_DIR = Path(__file__).with_name("frozen")  # tracked in git; tiny text files of image_ids
REAL_TEST_GENERATOR = "real_raise"


def _bucket(group: str) -> int:
    return int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100


def assign_splits(df: pd.DataFrame, holdout: str) -> pd.Series:
    """df needs: generator, group. Returns a split label per row."""
    def one(r) -> str:
        if r.generator in (holdout, REAL_TEST_GENERATOR):
            return "report_test"
        b = _bucket(r.group)
        return "train" if b < 70 else "val" if b < 85 else "gate"

    return df.apply(one, axis=1)


def digest(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def frozen_path() -> Path:
    return FROZEN_DIR / "report_test.txt"


def read_frozen() -> set[str] | None:
    """The frozen report_test ids, or None before the one-time freeze step."""
    p = frozen_path()
    if not p.exists():
        return None
    return {x for x in p.read_text().splitlines() if x and not x.startswith("#")}


def freeze(ids: list[str]) -> tuple[str, bool]:
    """Write the list once. If it exists it is never rewritten: returns (digest, False)."""
    if (old := read_frozen()) is not None:
        return digest(list(old)), False
    ids = sorted(set(ids))
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    frozen_path().write_text(f"# sha256 of sorted ids: {digest(ids)}\n" + "\n".join(ids) + "\n")
    return digest(ids), True
