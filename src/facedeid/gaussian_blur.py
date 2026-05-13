"""Gaussian Blurring de-identification (k×k kernel, OpenCV's auto-sigma)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from .face_detector import FaceDetector, BBox


def _normalize_ksize(k: int) -> int:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k % 2 == 0:
        k += 1
    return k


def gaussian_blur(img: np.ndarray, k: int, sigma: float = 0.0) -> np.ndarray:
    k = _normalize_ksize(k)
    if k == 1:
        return img.copy()
    return cv2.GaussianBlur(img, (k, k), sigmaX=float(sigma), sigmaY=float(sigma))


def gaussian_blur_region(
    img: np.ndarray, bbox: BBox, k: int, sigma: float = 0.0
) -> np.ndarray:
    out = img.copy()
    x, y, w, h = bbox
    H, W = img.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return out
    out[y0:y1, x0:x1] = gaussian_blur(img[y0:y1, x0:x1], k, sigma=sigma)
    return out


def gaussian_blur_faces(
    img: np.ndarray,
    k: int,
    detector: FaceDetector | None = None,
    fallback_full: bool = True,
    sigma: float = 0.0,
) -> tuple[np.ndarray, list[BBox]]:
    """Detect faces and blur each bbox. With fallback_full, treat whole image as bbox if none."""
    if detector is None:
        detector = FaceDetector(backend="haar")
    boxes = detector.detect(img, fallback_full=fallback_full)
    out = img.copy()
    for box in boxes:
        out = gaussian_blur_region(out, box, k, sigma=sigma)
    return out, boxes


# ---- CLI:批次處理整個資料夾 ------------------------------------------------
def _gather_images(root: Path) -> list[Path]:
    """遞迴蒐集 root 底下所有支援的影像檔。"""
    exts = {".pgm", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def _process_folder(
    in_root: Path,
    out_root: Path,
    k: int,
    sigma: float,
    detect_faces: bool,
    backend: str,
    quiet: bool = False,
) -> int:
    files = _gather_images(in_root)
    if not files:
        print(f"[gaussian_blur] no images under {in_root}", file=sys.stderr)
        return 0

    detector = FaceDetector(backend=backend) if detect_faces else None
    n = 0
    for f in files:
        rel = f.relative_to(in_root)
        # Always write PNG to avoid lossy re-encoding for downstream evaluation.
        out_path = out_root / rel.with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[gaussian_blur] skip unreadable {f}", file=sys.stderr)
            continue

        if detect_faces:
            out, _ = gaussian_blur_faces(img, k, detector=detector, fallback_full=True, sigma=sigma)
        else:
            out = gaussian_blur(img, k, sigma=sigma)

        cv2.imwrite(str(out_path), out)
        n += 1
        if not quiet and n % 50 == 0:
            print(f"  {n}/{len(files)}")
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--output", "-o", required=True)
    p.add_argument("--k", "-k", type=int, required=True)
    p.add_argument("--sigma", type=float, default=0.0)
    p.add_argument("--detect-faces", action="store_true")
    p.add_argument("--backend", choices=["haar", "hog"], default="haar")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    in_root = Path(args.input).resolve()
    out_root = Path(args.output).resolve()
    if not in_root.is_dir():
        p.error(f"input not a directory: {in_root}")

    eff_k = _normalize_ksize(args.k)
    print(f"[gaussian_blur] k={eff_k}  sigma={args.sigma or 'auto'}  detect_faces={args.detect_faces}")
    print(f"  in : {in_root}")
    print(f"  out: {out_root}")
    n = _process_folder(in_root, out_root, args.k, args.sigma, args.detect_faces, args.backend, args.quiet)
    print(f"[gaussian_blur] done: {n} images")


if __name__ == "__main__":
    main()
