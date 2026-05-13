"""DP-Blur mechanisms aligned with Step 1 Gaussian Blur (k = 15/45/99).

Two mechanisms are provided so that the report can contrast the privacy-utility
trade-off:

* lp_blur (Laplace-Perturbed Blur)
    Apply k x k Gaussian blur, then add Laplace(0, 255 / (k**2 * eps)) noise to
    every output pixel. This treats sensitivity as 255 / k**2 (the cell-mean
    sensitivity from Fan 2018 DP-Pix) and uses ONE budget eps for the whole
    image. It does NOT account for the fact that a single input pixel affects
    k**2 output pixels through the kernel, so it does NOT strictly satisfy
    epsilon-DP. Useful as a "blur + noise" reference baseline.

* dp_blur_split (Budget-Split Differentially Private Blur)
    Apply k x k Gaussian blur. To strictly satisfy epsilon-DP we use sequential
    composition: each input pixel can affect up to k**2 outputs, so we allocate
    budget eps / k**2 per output pixel. With sensitivity 255 per output (the
    max change from a single full-range pixel flip) the Laplace scale becomes
    255 * k**2 / eps. This is a conservative bound on the true per-output
    sensitivity (~ max_weight * 255 ~= 255 / k**2 for a Gaussian kernel), so
    the noise added is larger than strictly necessary. Strict epsilon-DP is
    therefore preserved; utility suffers correspondingly.

The CLI generates one (k, eps) dataset directory at a time and mirrors the
input subject/file structure (writing PNG output).
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


def lp_blur(
    img: np.ndarray,
    k: int,
    epsilon: float,
    sigma: float = 0.0,
) -> np.ndarray:
    """LP-Blur: blur then add Laplace(0, 255/(k**2 * eps)) per pixel.

    Not strict epsilon-DP (no composition over the k**2 affected outputs).
    """
    k = _normalize_ksize(k)
    img_f = img.astype(np.float64)
    blurred = cv2.GaussianBlur(img_f, (k, k), sigmaX=float(sigma), sigmaY=float(sigma))
    scale = 255.0 / (k * k * epsilon)
    noise = np.random.laplace(loc=0.0, scale=scale, size=blurred.shape)
    out = np.clip(blurred + noise, 0.0, 255.0)
    return out.astype(np.uint8)


def dp_blur_split(
    img: np.ndarray,
    k: int,
    epsilon: float,
    sigma: float = 0.0,
) -> np.ndarray:
    """DP-Blur-Split: budget eps split across the k**2 affected outputs.

    Per-output Laplace scale is 255 * k**2 / epsilon. Strictly epsilon-DP under
    the conservative sensitivity bound (255 per output).
    """
    k = _normalize_ksize(k)
    img_f = img.astype(np.float64)
    blurred = cv2.GaussianBlur(img_f, (k, k), sigmaX=float(sigma), sigmaY=float(sigma))
    scale = 255.0 * (k * k) / epsilon
    noise = np.random.laplace(loc=0.0, scale=scale, size=blurred.shape)
    out = np.clip(blurred + noise, 0.0, 255.0)
    return out.astype(np.uint8)


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
    if mechanism not in MECHANISMS:
        raise ValueError(f"Unknown mechanism {mechanism!r}. Choices: {list(MECHANISMS)}")
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
    parser = argparse.ArgumentParser(description="Generate one DP-Blur dataset variant.")
    parser.add_argument("--src", default="data/att_faces", help="Input dataset root.")
    parser.add_argument(
        "--dst",
        required=True,
        help="Output dataset root (created if missing).",
    )
    parser.add_argument("--mechanism", choices=list(MECHANISMS), required=True)
    parser.add_argument("--k", type=int, required=True, help="Gaussian kernel size.")
    parser.add_argument("--eps", type=float, required=True, help="Privacy budget epsilon.")
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.0,
        help="Gaussian sigma. 0 = OpenCV auto from k (default).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional NumPy seed for reproducible noise.",
    )
    args = parser.parse_args()

    n = process_dataset(
        input_dir=args.src,
        output_dir=args.dst,
        mechanism=args.mechanism,
        k=args.k,
        epsilon=args.eps,
        sigma=args.sigma,
        seed=args.seed,
    )
    print(f"mechanism={args.mechanism} k={args.k} eps={args.eps} -> wrote {n} images to {args.dst}")


if __name__ == "__main__":
    main()
