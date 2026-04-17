"""
Memory / state persistence: save and load pipeline intermediate results.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def save_heightfield(heightfield: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), heightfield)


def load_heightfield(path: str | Path) -> np.ndarray:
    return np.load(str(path))
