"""
PairedTextureDataset: loads (diffuse, height) pairs from a flat, preprocessed
dataset produced by scripts/preprocess_dataset.py.

Layout:
    <data_root>/diffuse/*.png
    <data_root>/height/*.png

Pairing is by sorted-order alignment — the two folders must contain the same
number of files with matching basenames.

Augmentations (train=True) are purely geometric and fully synced across the
diffuse/height pair: horizontal flip, vertical flip, 90° rotation. No colour
jitter (texture semantics must stay intact).
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from cnc_params import IMAGE_PX


class PairedTextureDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        image_size: int = IMAGE_PX,
        train: bool = True,
    ) -> None:
        self.root = Path(data_root)
        self.image_size = image_size
        self.train = train

        diff_dir = self.root / "diffuse"
        height_dir = self.root / "height"
        assert diff_dir.is_dir(), f"Missing diffuse dir: {diff_dir}"
        assert height_dir.is_dir(), f"Missing height dir: {height_dir}"

        self.diffuse_files = sorted(diff_dir.glob("*.png"))
        self.height_files = sorted(height_dir.glob("*.png"))
        assert len(self.diffuse_files) == len(self.height_files), (
            f"Pair count mismatch: {len(self.diffuse_files)} diffuse "
            f"vs {len(self.height_files)} height"
        )
        assert len(self.diffuse_files) > 0, f"No .png files under {self.root}"

    def __len__(self) -> int:
        return len(self.diffuse_files)

    def _load(self, d_path: Path, h_path: Path) -> tuple[np.ndarray, np.ndarray]:
        diffuse_bgr = cv2.imread(str(d_path), cv2.IMREAD_COLOR)
        if diffuse_bgr is None:
            raise RuntimeError(f"Failed to read {d_path}")
        diffuse = cv2.cvtColor(diffuse_bgr, cv2.COLOR_BGR2RGB)
        height = cv2.imread(str(h_path), cv2.IMREAD_GRAYSCALE)
        if height is None:
            raise RuntimeError(f"Failed to read {h_path}")

        s = self.image_size
        if diffuse.shape[:2] != (s, s):
            diffuse = cv2.resize(diffuse, (s, s), interpolation=cv2.INTER_AREA)
        if height.shape[:2] != (s, s):
            height = cv2.resize(height, (s, s), interpolation=cv2.INTER_AREA)
        return diffuse, height

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        d_path = self.diffuse_files[idx]
        h_path = self.height_files[idx]
        diffuse, height = self._load(d_path, h_path)

        if self.train:
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

        diffuse_t = torch.from_numpy(diffuse).permute(2, 0, 1).float() / 127.5 - 1.0
        height_t = torch.from_numpy(height).unsqueeze(0).float() / 127.5 - 1.0
        return diffuse_t, height_t
