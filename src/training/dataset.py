"""
PairedTextureDataset: loads (diffuse, height) pairs from the resized dataset.

Directory layout expected (any subfolder under root):
    root/<category>/<sample_name>/<sample_name>_diffuse.{png,jpg}
    root/<category>/<sample_name>/<sample_name>_height.{png,jpg}

Augmentations (train=True):
  - random crop to image_size
  - random horizontal/vertical flip (paired)
  - random 90 deg rotation (paired)
  - diffuse-only: brightness/contrast jitter, mild Gaussian noise
  - height-only: mild gamma jitter to vary relief

All geometric transforms are applied identically to diffuse and height so the
two stay pixel-aligned.
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


SUPPORTED_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _find_pairs(root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for sample_dir in sorted(root.rglob("*")):
        if not sample_dir.is_dir():
            continue
        diffuse = None
        height = None
        for f in sample_dir.iterdir():
            name = f.name.lower()
            if not name.endswith(SUPPORTED_EXT):
                continue
            if "_diffuse" in name:
                diffuse = f
            elif "_height" in name:
                height = f
        if diffuse is not None and height is not None:
            pairs.append((diffuse, height))
    return pairs


class PairedTextureDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        image_size: int = 256,
        train: bool = True,
        augment: bool = True,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.train = train
        self.augment = augment and train
        self.samples = _find_pairs(self.root)
        if len(self.samples) == 0:
            raise RuntimeError(f"No (diffuse, height) pairs found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, d_path: Path, h_path: Path) -> tuple[np.ndarray, np.ndarray]:
        diffuse_bgr = cv2.imread(str(d_path), cv2.IMREAD_COLOR)
        if diffuse_bgr is None:
            raise RuntimeError(f"Failed to read {d_path}")
        diffuse = cv2.cvtColor(diffuse_bgr, cv2.COLOR_BGR2RGB)
        height = cv2.imread(str(h_path), cv2.IMREAD_GRAYSCALE)
        if height is None:
            raise RuntimeError(f"Failed to read {h_path}")
        if diffuse.shape[:2] != height.shape[:2]:
            s = min(diffuse.shape[0], height.shape[0])
            diffuse = cv2.resize(diffuse, (s, s), interpolation=cv2.INTER_AREA)
            height = cv2.resize(height, (s, s), interpolation=cv2.INTER_AREA)
        return diffuse, height

    def _random_crop(self, diffuse: np.ndarray, height: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = diffuse.shape[:2]
        size = self.image_size
        if h < size or w < size:
            diffuse = cv2.resize(diffuse, (size, size), interpolation=cv2.INTER_AREA)
            height = cv2.resize(height, (size, size), interpolation=cv2.INTER_AREA)
            return diffuse, height
        y = random.randint(0, h - size)
        x = random.randint(0, w - size)
        return diffuse[y:y + size, x:x + size], height[y:y + size, x:x + size]

    def _center_crop(self, diffuse: np.ndarray, height: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = diffuse.shape[:2]
        size = self.image_size
        if h < size or w < size:
            diffuse = cv2.resize(diffuse, (size, size), interpolation=cv2.INTER_AREA)
            height = cv2.resize(height, (size, size), interpolation=cv2.INTER_AREA)
            return diffuse, height
        y = (h - size) // 2
        x = (w - size) // 2
        return diffuse[y:y + size, x:x + size], height[y:y + size, x:x + size]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        d_path, h_path = self.samples[idx]
        diffuse, height = self._load(d_path, h_path)

        if self.augment:
            diffuse, height = self._random_crop(diffuse, height)

            if random.random() < 0.5:
                diffuse = np.ascontiguousarray(diffuse[:, ::-1])
                height = np.ascontiguousarray(height[:, ::-1])
            if random.random() < 0.5:
                diffuse = np.ascontiguousarray(diffuse[::-1, :])
                height = np.ascontiguousarray(height[::-1, :])
            if random.random() < 0.75:
                k = random.randint(1, 3)
                diffuse = np.ascontiguousarray(np.rot90(diffuse, k))
                height = np.ascontiguousarray(np.rot90(height, k))

            if random.random() < 0.5:
                d = diffuse.astype(np.float32)
                d = d * random.uniform(0.85, 1.15)
                d = (d - 128.0) * random.uniform(0.9, 1.1) + 128.0
                diffuse = np.clip(d, 0, 255).astype(np.uint8)
            if random.random() < 0.3:
                noise = np.random.normal(0, 3.0, diffuse.shape).astype(np.float32)
                diffuse = np.clip(diffuse.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            if random.random() < 0.3:
                gamma = random.uniform(0.85, 1.15)
                h01 = height.astype(np.float32) / 255.0
                h01 = np.power(h01, gamma)
                height = np.clip(h01 * 255.0, 0, 255).astype(np.uint8)
        else:
            diffuse, height = self._center_crop(diffuse, height)

        diffuse_t = torch.from_numpy(diffuse).permute(2, 0, 1).float() / 127.5 - 1.0
        height_t = torch.from_numpy(height).unsqueeze(0).float() / 127.5 - 1.0
        return diffuse_t, height_t
