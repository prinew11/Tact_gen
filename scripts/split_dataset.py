"""
将 dataset 中的图片按 原图(diffuse) 和 heightmap 分成两个大文件夹。

原始结构:
    dataset/Bark/bark 01/bark 01_diffuse.png
    dataset/Bark/bark 01/bark 01_height.png
    dataset/Wood/xxx/xxx_diffuse.jpg
    ...

输出结构:
    dataset_split/original/Bark__bark_01_diffuse.png
    dataset_split/heightmap/Bark__bark_01_height.png
    ...

用法:
    python scripts/split_dataset.py
    python scripts/split_dataset.py --src D:/other/dataset --dst D:/other/dataset_split
"""

import argparse
import shutil
from pathlib import Path

DATASET_DIR = Path(r"D:\homework\lund\CS_project\dataset")
OUTPUT_DIR = Path(r"D:\homework\lund\CS_project\dataset_split")
CATEGORIES = ["Bark", "Wood", "Flooring"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default=str(DATASET_DIR))
    parser.add_argument("--dst", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    orig_dir = dst / "original"
    hmap_dir = dst / "heightmap"
    orig_dir.mkdir(parents=True, exist_ok=True)
    hmap_dir.mkdir(parents=True, exist_ok=True)

    n_orig, n_hmap = 0, 0

    for cat in CATEGORIES:
        cat_dir = src / cat
        if not cat_dir.is_dir():
            print(f"[SKIP] {cat_dir} not found")
            continue

        for sub in sorted(cat_dir.iterdir()):
            if not sub.is_dir():
                continue

            for f in sorted(sub.iterdir()):
                if not f.is_file():
                    continue

                name_lower = f.name.lower()
                safe_name = f"{cat}__{f.name}"

                if "_height" in name_lower:
                    shutil.copy2(f, hmap_dir / safe_name)
                    n_hmap += 1
                elif "_diffuse" in name_lower or "_color" in name_lower or "_albedo" in name_lower:
                    shutil.copy2(f, orig_dir / safe_name)
                    n_orig += 1

    print(f"Done.")
    print(f"  original  -> {orig_dir}  ({n_orig} files)")
    print(f"  heightmap -> {hmap_dir}  ({n_hmap} files)")


if __name__ == "__main__":
    main()
