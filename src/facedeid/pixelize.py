"""Pixelization: replace each b×b cell with its mean (downsample + upsample equivalent)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from .face_detector import FaceDetector, BBox


def pixelize(img: np.ndarray, b: int) -> np.ndarray:
    if b < 1:
        raise ValueError(f"b must be >= 1, got {b}")
    if b == 1:
        return img.copy()
    H, W = img.shape[:2]
    # INTER_AREA is the per-cell mean; INTER_NEAREST then expands each pixel to b×b.
    small = cv2.resize(img, (max(1, W // b), max(1, H // b)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)


def pixelize_region(img: np.ndarray, bbox: BBox, b: int) -> np.ndarray:
    out = img.copy()
    x, y, w, h = bbox
    H, W = img.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return out
    out[y0:y1, x0:x1] = pixelize(img[y0:y1, x0:x1], b)
    return out


def pixelize_faces(
    img: np.ndarray,
    b: int,
    detector: FaceDetector | None = None,
    fallback_full: bool = True,
) -> tuple[np.ndarray, list[BBox]]:
    """Detect faces and pixelize each bbox. With fallback_full, treat whole image as bbox if none."""
    if detector is None:
        detector = FaceDetector(backend="haar")
    boxes = detector.detect(img, fallback_full=fallback_full)
    out = img.copy()
    for box in boxes:
        out = pixelize_region(out, box, b)
    return out, boxes


def _gather_images(root: Path) -> list[Path]:
    exts = {".pgm", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def _process_folder(
    in_root: Path,
    out_root: Path,
    b: int,
    detect_faces: bool,
    backend: str,
    quiet: bool = False,
) -> int:
    files = _gather_images(in_root)
    if not files:
        print(f"[pixelize] no images under {in_root}", file=sys.stderr)
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
            print(f"[pixelize] skip unreadable {f}", file=sys.stderr)
            continue

        if detect_faces:
            out, _ = pixelize_faces(img, b, detector=detector, fallback_full=True)
        else:
            out = pixelize(img, b)

        cv2.imwrite(str(out_path), out)
        n += 1
        if not quiet and n % 50 == 0:
            print(f"  {n}/{len(files)}")
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--output", "-o", required=True)
    p.add_argument("--b", "-b", type=int, required=True)
    p.add_argument("--detect-faces", action="store_true")
    p.add_argument("--backend", choices=["haar", "hog"], default="haar")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    in_root = Path(args.input).resolve()
    out_root = Path(args.output).resolve()
    if not in_root.is_dir():
        p.error(f"input not a directory: {in_root}")

    print(f"[pixelize] b={args.b}  detect_faces={args.detect_faces}")
    print(f"  in : {in_root}")
    print(f"  out: {out_root}")
    n = _process_folder(in_root, out_root, args.b, args.detect_faces, args.backend, args.quiet)
    print(f"[pixelize] done: {n} images")


if __name__ == "__main__":
    main()
