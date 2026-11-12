"""Cấu hình tập trung: đọc configs/*.yaml + biến môi trường (.env)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]          # .../diabetes_mas
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
CASES_DIR = DATA_DIR / "cases"
CACHE_DIR = DATA_DIR / "cache"
KB_DIR = PROJECT_ROOT / "kb"
SEED_DIR = KB_DIR / "seed"


def _load_dotenv(path: Path) -> None:
    """Nạp .env tối giản (KEY=VALUE), không ghi đè biến đã có."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv(PROJECT_ROOT / ".env")


@lru_cache(maxsize=None)
def load_yaml(name: str) -> dict[str, Any]:
    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def settings() -> dict[str, Any]:
    return load_yaml("settings.yaml")


def env(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)
