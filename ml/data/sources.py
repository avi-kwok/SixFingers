"""Source registry (sources.yaml) and the per-source raw index (data/raw/<name>/index.jsonl)."""
import json
from pathlib import Path

import yaml

from ml.common.paths import raw_dir

REGISTRY = Path(__file__).with_name("sources.yaml")


def load_registry() -> dict:
    return yaml.safe_load(REGISTRY.read_text())


def enabled_sources(reg: dict | None = None) -> list[dict]:
    return [s for s in (reg or load_registry())["sources"] if s.get("enabled")]


def holdout_generator(reg: dict | None = None) -> str:
    return (reg or load_registry())["holdout_generator"]


def source_dir(name: str) -> Path:
    return raw_dir() / name


def read_index(name: str) -> list[dict]:
    p = source_dir(name) / "index.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def append_index(name: str, row: dict) -> None:
    p = source_dir(name) / "index.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(row) + "\n")
