"""Where things live. Override with SIXFINGERS_DATA_DIR / SIXFINGERS_MODELS_DIR."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return Path(os.environ.get("SIXFINGERS_DATA_DIR", REPO_ROOT / "data"))


def models_dir() -> Path:
    return Path(os.environ.get("SIXFINGERS_MODELS_DIR", REPO_ROOT / "models"))


def raw_dir() -> Path: return data_dir() / "raw"
def normalized_dir() -> Path: return data_dir() / "normalized"
def features_dir() -> Path: return data_dir() / "features"
def manifest_path() -> Path: return data_dir() / "manifest.parquet"
def current_model_dir() -> Path: return models_dir() / "current"  # symlink -> vN
def model_version_dir(version: str) -> Path: return models_dir() / version
