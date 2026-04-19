"""Utilities module for loading configuration."""

from typing import Any

import yaml


def load_config(path: str) -> dict[str, Any]:
    """Load YAML configuration file and return as dictionary."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config
