"""Load credentials from data.env (never commit data.env)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_env_file() -> None:
    for name in ("data.env", ".env.local"):
        path = ROOT / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key, val = key.strip(), val.strip().strip('"').strip("'")
            # Always apply data.env (allows updating keys in same file)
            os.environ[key] = val
        break


def get_env(key: str, default: str | None = None) -> str | None:
    load_env_file()
    return os.environ.get(key, default)


def configure_hf_mirror() -> str | None:
    """Use hf-mirror.com when HF_ENDPOINT is unset (common in CN)."""
    load_env_file()
    endpoint = os.environ.get("HF_ENDPOINT")
    if endpoint:
        return endpoint
    mirror = "https://hf-mirror.com"
    os.environ["HF_ENDPOINT"] = mirror
    return mirror
