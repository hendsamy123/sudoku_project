"""Reproducible MNIST download + preprocessing (digits 1-9 only).

Run:  python -m data.make_data
Output: data/mnist_1to9.npz  (uint8 images, int labels 1..9)

Steps
1. Download the four original MNIST gzip files (tries the original LeCun page,
   then two mirrors that serve byte-identical files). A local copy is kept in data/raw/.
2. Drop digit 0 (Sudoku has only 1-9).
3. Split the ORIGINAL training set into train / validation with a fixed seed
   (10,000 validation images). The original 10k test set is left untouched.
"""
import gzip
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "mnist_1to9.npz"
SEED = 7
N_VAL = 10_000

MIRRORS = [
    "http://yann.lecun.com/exdb/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://raw.githubusercontent.com/golbin/TensorFlow-MNIST/master/mnist/data/",
]
FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
         "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
EXPECTED = {"train-images-idx3-ubyte.gz": (2051, 60000), "train-labels-idx1-ubyte.gz": (2049, 60000),
            "t10k-images-idx3-ubyte.gz": (2051, 10000), "t10k-labels-idx1-ubyte.gz": (2049, 10000)}


def _parse(path: Path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic, n = struct.unpack(">II", data[:8])
    if EXPECTED[path.name] != (magic, n):
        raise ValueError(f"{path.name}: unexpected header {(magic, n)}")
    if magic == 2051:
        rows, cols = struct.unpack(">II", data[8:16])
        return np.frombuffer(data, np.uint8, offset=16).reshape(n, rows, cols)
    return np.frombuffer(data, np.uint8, offset=8)


def download(name: str) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / name
    if path.exists():
        try:
            _parse(path)
            return path
        except Exception:
            path.unlink()
    for base in MIRRORS:
        try:
            print(f"downloading {base}{name}")
            urllib.request.urlretrieve(base + name, path)
            _parse(path)
            return path
        except Exception as e:  # try next mirror
            print("  failed:", e)
            if path.exists():
                path.unlink()
    sys.exit(f"Could not download {name}. Put the four MNIST .gz files in {RAW} manually.")


def main():
    arrays = {n: _parse(download(n)) for n in FILES}
    xtr, ytr = arrays[FILES[0]], arrays[FILES[1]].astype(np.int64)
    xte, yte = arrays[FILES[2]], arrays[FILES[3]].astype(np.int64)
    keep_tr, keep_te = ytr != 0, yte != 0
    xtr, ytr, xte, yte = xtr[keep_tr], ytr[keep_tr], xte[keep_te], yte[keep_te]
    perm = np.random.default_rng(SEED).permutation(len(xtr))
    val_idx, tr_idx = perm[:N_VAL], perm[N_VAL:]
    np.savez_compressed(OUT, x_train=xtr[tr_idx], y_train=ytr[tr_idx], x_val=xtr[val_idx], y_val=ytr[val_idx],
                        x_test=xte, y_test=yte)
    print(f"saved {OUT}: train={len(tr_idx)} val={len(val_idx)} test={len(xte)} (digits 1-9, seed={SEED})")


def load_data():
    """Returns dict of float32 images in [0,1] shaped (N,784) plus labels 1..9."""
    if not OUT.exists():
        raise FileNotFoundError("Run `python -m data.make_data` first.")
    d = np.load(OUT)
    out = {}
    for k in ("train", "val", "test"):
        out["x_" + k] = d["x_" + k].reshape(len(d["x_" + k]), -1).astype(np.float32) / 255.0
        out["y_" + k] = d["y_" + k]
    return out


if __name__ == "__main__":
    main()
