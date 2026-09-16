"""Paths and other repo-wide configuration.

Paths are resolved relative to this file so the repo works regardless of
where it is cloned. Override individual entries here if you store the
downloaded model weights elsewhere.
"""
from pathlib import Path

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

model_weights_path = {
    'echotracker': str(WEIGHTS_DIR / 'echotracker'),
    'pips++': str(WEIGHTS_DIR / 'pips2'),
    'cotracker3': str(WEIGHTS_DIR / 'cotracker3'),
    'specknet': str(WEIGHTS_DIR / 'specknet'),
    'locotrack': str(WEIGHTS_DIR / 'locotrack'),
    'echotracker2': str(WEIGHTS_DIR / 'echotracker2'),
}

