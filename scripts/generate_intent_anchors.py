"""
Generate intent anchor vectors from sweep_mapping.csv.

For each intent (rough, soft, hard, organic), selects rows matching
specific parameter combinations and computes mean GLCM features.

Usage:
    python scripts/generate_intent_anchors.py

Output: data/intent_anchors.json
"""
import csv
import json
from pathlib import Path

import numpy as np

SWEEP_CSV = Path("outputs/feature_analysis/sweep_mapping.csv")
OUTPUT_JSON = Path("data/intent_anchors.json")

# GLCM feature columns (9-dim input to MLP)
GLCM_COLS = [
    "contrast", "homogeneity", "energy", "correlation", "entropy",
    "height_mean", "height_std", "hist_entropy", "gradient_mean",
]

# Intent definitions: parameter filters → which rows to average
INTENT_FILTERS = {
    "rough": {
        "morph_strength": (0.4, 0.6),   # == 0.5
        "gamma": (1.9, 2.1),            # == 2.0
    },
    "soft": {
        "morph_strength": (1.9, 2.1),   # == 2.0
        "gamma": (0.4, 0.6),            # == 0.5
        "edge_sigma": (4.0, 5.0),       # == 4.5
    },
    "hard": {
        "morph_strength": (1.9, 2.1),   # == 2.0
        "gamma": (0.4, 0.6),            # == 0.5
        "edge_sigma": (0.4, 0.6),       # == 0.5
    },
    "organic": {
        "morph_strength": (1.9, 2.1),   # == 2.0
        "gamma": (1.4, 1.6),            # == 1.5
    },
}

# Recommended terrace_steps per intent
INTENT_STEPS = {
    "rough":   10,
    "soft":    6,
    "hard":    4,
    "organic": 8,
}


def load_sweep(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def filter_rows(rows: list[dict], filters: dict) -> list[dict]:
    matched = []
    for row in rows:
        ok = True
        for col, (lo, hi) in filters.items():
            val = float(row[col])
            if val < lo or val > hi:
                ok = False
                break
        if ok:
            matched.append(row)
    return matched


def mean_glcm(rows: list[dict]) -> list[float]:
    vals = np.array([[float(r[c]) for c in GLCM_COLS] for r in rows])
    return vals.mean(axis=0).tolist()


def main():
    rows = load_sweep(SWEEP_CSV)
    print(f"Loaded {len(rows)} rows from {SWEEP_CSV}")

    anchors = {}
    for intent, filters in INTENT_FILTERS.items():
        matched = filter_rows(rows, filters)
        if not matched:
            print(f"  [WARN] {intent}: no rows matched filters {filters}")
            continue
        vec = mean_glcm(matched)
        anchors[intent] = vec
        print(f"  {intent:10s}: {len(matched)} rows")

    output = {
        "intents": anchors,
        "terrace_steps": INTENT_STEPS,
        "feature_cols": GLCM_COLS,
        "description": "Mean GLCM feature vectors for each tactile intent.",
        "interpolation": "target = alpha * intent_a + (1-alpha) * intent_b",
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {OUTPUT_JSON}")

    # Verification: print feature comparison table
    print("\n=== Anchor feature comparison ===")
    print(f"{'Feature':<16}" + "".join(f"{k:>10}" for k in anchors))
    for i, col in enumerate(GLCM_COLS):
        row_str = f"{col:<16}"
        for k in anchors:
            row_str += f"{anchors[k][i]:>10.3f}"
        print(row_str)
    print(f"\n{'terrace_steps':<16}" + "".join(f"{INTENT_STEPS[k]:>10}" for k in anchors))


if __name__ == "__main__":
    main()
