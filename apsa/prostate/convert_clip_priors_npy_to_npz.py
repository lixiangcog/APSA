"""
convert_clip_priors_npy_to_npz.py

Why:
  Some CLIP priors in this repo are stored as pickled dicts via `np.save(..., dict)`.
  If those files were created with a newer NumPy, older environments (e.g. NumPy 1.19 in `pass`)
  may fail to unpickle them (e.g. `No module named 'numpy._core'`).

What:
  Convert `*_clip.npy` (pickled dict) -> `*_clip.npz` (non-pickled arrays),
  so they are loadable across NumPy versions.
"""

import argparse
import os
from pathlib import Path

import numpy as np


def convert_one(npy_path: Path, overwrite: bool) -> bool:
    if not npy_path.name.endswith("_clip.npy"):
        return False

    npz_path = npy_path.with_suffix(".npz")
    if npz_path.exists() and not overwrite:
        return False

    data = np.load(str(npy_path), allow_pickle=True).item()
    soft_map = data.get("soft_map", None)
    if soft_map is None:
        return False

    out = {"soft_map": np.asarray(soft_map)}
    if "slice_idx" in data:
        out["slice_idx"] = np.asarray(int(data["slice_idx"]), dtype=np.int32)

    np.savez_compressed(str(npz_path), **out)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/MRI_prostate/clip_priors", help="Root folder containing domain subfolders.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .npz outputs.")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    converted = 0
    scanned = 0
    for npy_path in root.rglob("*_clip.npy"):
        scanned += 1
        try:
            if convert_one(npy_path, overwrite=args.overwrite):
                converted += 1
        except Exception as e:
            print(f"[WARN] {npy_path}: {e}")

    print(f"Scanned: {scanned}  Converted: {converted}  Root: {root}")


if __name__ == "__main__":
    main()

