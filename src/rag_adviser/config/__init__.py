"""Configuration loading for ragadvisor."""

from __future__ import annotations

from pathlib import Path

import yaml

_CONFIG_DIR = Path(__file__).parent


def load_defaults() -> dict:
    """Load default configuration from defaults.yaml."""
    defaults_path = _CONFIG_DIR / "defaults.yaml"
    with open(defaults_path, encoding="utf-8") as f:
        return yaml.safe_load(f)
