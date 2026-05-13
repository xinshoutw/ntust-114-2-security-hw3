"""DP-Pixelization: per-cell mean with Laplace noise (Fan 2018, m=1)."""

import argparse
from pathlib import Path

import cv2
import numpy as np


def dp_pixelization(img: np.ndarray, block_size: int, epsilon: float) -> np.ndarray:
    h, w = img.shape
    img_f = img.astype(np.float64)
    out = img_f.copy()
    scale = 255.0 / (block_size ** 2) / epsilon

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            mean = img_f[i:i + block_size, j:j + block_size].mean()
            out[i:i + block_size, j:j + block_size] = mean + np.random.laplace(0.0, scale)

    return np.clip(out, 0, 255).astype(np.uint8)


def process_dataset(
    input_dir: str,
    output_dir: str,
    block_size: int,
    epsilon: float,
    seed: int | None = None,
):
    if seed is not None:
        np.random.seed(seed)

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for pgm in sorted(input_path.rglob("*.pgm")):
        img = cv2.imread(str(pgm), cv2.IMREAD_GRAYSCALE)
        out = dp_pixelization(img, block_size, epsilon)
        rel = pgm.relative_to(input_path)
        dst = output_path / rel.with_suffix(".png")
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/att_faces")
    parser.add_argument("--dst", default="output/dp_pix")
    parser.add_argument("--block", type=int, default=8)
    parser.add_argument("--eps", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    out_dir = f"{args.dst}_b{args.block}/eps{args.eps}"
    process_dataset(args.src, out_dir, args.block, args.eps, seed=args.seed)
    print(f"Done → {out_dir}")
