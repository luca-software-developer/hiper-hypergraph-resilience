# -*- coding: utf-8 -*-
"""
config.py

Loads project configuration from a JSON file.
"""

import json
from pathlib import Path

CONFIG_FILE = Path(__file__).parent.parent / "config.json"


def load_config() -> dict:
    """
    Load and return the project configuration from a JSON file.

    Returns:
        dict: Configuration parameters (e.g., dataset path and name).
    """
    with open(CONFIG_FILE) as f:
        return json.load(f)
