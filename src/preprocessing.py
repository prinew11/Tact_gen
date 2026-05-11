"""
Visual preprocessing: grayscale, edge detection, high-frequency content.
"""
from __future__ import annotations

import cv2
import numpy as np


def load_image_gray(path: str) -> np.ndarray:
    """Load image and convert to grayscale float32 in [0, 1]."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img.astype(np.float32) / 255.0


def load_image_rgb(path: str) -> np.ndarray:
    """Load image as RGB float32 in [0, 1]."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def extract_edges(gray: np.ndarray) -> np.ndarray:
    """Otsu-tuned Canny edge map normalized to [0, 1]."""
    uint8 = (gray * 255).astype(np.uint8)
    otsu_val, _ = cv2.threshold(uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(uint8, threshold1=otsu_val * 0.5, threshold2=otsu_val * 1.0)
    return edges.astype(np.float32) / 255.0


def extract_frequency(gray: np.ndarray) -> np.ndarray:
    """High-frequency content via |Laplacian|, normalized to [0, 1]."""
    uint8 = (gray * 255).astype(np.uint8)
    lap = np.abs(cv2.Laplacian(uint8, cv2.CV_32F))
    max_val = lap.max()
    if max_val > 0:
        lap /= max_val
    return lap
