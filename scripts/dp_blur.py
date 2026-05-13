"""Two DP-Blur mechanisms (k = 15/45/99) for the Step 3 sweep.

lp_blur:        per-pixel Laplace scale = 255 / (k**2 * eps). No composition,
                not strict epsilon-DP — reference baseline only.
dp_blur_split:  per-pixel Laplace scale = 255 * k**2 / eps. Budget split across
                the k**2 outputs a single input affects (sequential composition).
                Strict epsilon-DP under a 255-per-output sensitivity bound.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


IMG_EXTS = {".pgm", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _normalize_ksize(k: int) -> int:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k % 2 == 0:
        k += 1
    return k


def _blur_then_noise(img: np.ndarray, k: int, sigma: float, scale: float) -> np.ndarray:
    img_f = img.astype(np.float64)
    blurred = cv2.GaussianBlur(img_f, (k, k), sigmaX=float(sigma), sigmaY=float(sigma))
    noise = np.random.laplace(loc=0.0, scale=scale, size=blurred.shape)
    return np.clip(blurred + noise, 0.0, 255.0).astype(np.uint8)


def lp_blur(img: np.ndarray, k: int, epsilon: float, sigma: float = 0.0) -> np.ndarray:
    k = _normalize_ksize(k)
    return _blur_then_noise(img, k, sigma, scale=255.0 / (k * k * epsilon))


def dp_blur_split(img: np.ndarray, k: int, epsilon: float, sigma: float = 0.0) -> np.ndarray:
    k = _normalize_ksize(k)
    return _blur_then_noise(img, k, sigma, scale=255.0 * (k * k) / epsilon)


MECHANISMS = {
    "lp": lp_blur,
    "split": dp_blur_split,
}


def process_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    mechanism: str,
    k: int,
    epsilon: float,
    sigma: float = 0.0,
    seed: int | None = None,
) -> int:
    fn = MECHANISMS[mechanism]
    if seed is not None:
        np.random.seed(seed)

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    n = 0
    for src in sorted(input_path.rglob("*")):
        if not src.is_file() or src.suffix.lower() not in IMG_EXTS:
            continue
        img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        out = fn(img, k, epsilon, sigma)
        rel = src.relative_to(input_path)
        dst = output_path / rel.with_suffix(".png")
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), out)
        n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/att_faces")
    parser.add_argument("--dst", required=True)
    parser.add_argument("--mechanism", choices=list(MECHANISMS), required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--eps", type=float, required=True)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    n = process_dataset(
        args.src, args.dst, args.mechanism, args.k, args.eps, args.sigma, args.seed
    )
    print(f"{args.mechanism} k={args.k} eps={args.eps} -> {n} images at {args.dst}")


if __name__ == "__main__":
    main()
