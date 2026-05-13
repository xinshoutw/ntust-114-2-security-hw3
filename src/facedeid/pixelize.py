"""
pixelize.py
-----------
Pixelization 主程式。

演算法(對應作業 PDF 第 6 頁):
    For each cell of b×b pixels, replace all pixels in the cell with
    the cell's mean. Smaller b → 較好視覺品質。

實作走「下採樣再上採樣」的等價形式,效率比逐 cell 取平均高:
    1. resize down 到 (W // b, H // b),用 INTER_AREA(等於每個 cell 取平均)
    2. resize up 回 (W, H),用 INTER_NEAREST(把每個 cell 還原成 b×b 純色)

支援兩種使用方式:
    * 對「全圖」做 pixelization:適合 ORL 這種 cropped 人臉
    * 對「人臉 bbox 區域」做 pixelization:適合 FaceScrub / 自然影像
      (用 face_detector 抓 bbox,只把 bbox 區域 pixelize 後貼回原圖)

用法:
    # 程式碼
    from facedeid.pixelize import pixelize, pixelize_faces
    out = pixelize(img, b=8)

    # CLI:批次處理整個資料夾(在專案根目錄用 uv run)
    uv run python -m facedeid.pixelize --input data/att_faces --output data/deid/pixelized/pix_b8 --b 8
    uv run python -m facedeid.pixelize --input data/celeb --output data/deid/celeb_pix_b8 --b 8 --detect-faces
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from .face_detector import FaceDetector, BBox


# ---- 核心演算法 -------------------------------------------------------------
def pixelize(img: np.ndarray, b: int) -> np.ndarray:
    """
    對整張影像做 b×b pixelization。

    參數:
      img: numpy array,(H, W) 灰階 或 (H, W, 3) BGR
      b  : pixel block 邊長(必須 ≥ 1)

    回傳: 與 img 同 shape 同 dtype 的 pixelized 影像。
    """
    if b < 1:
        raise ValueError(f"b 必須 ≥ 1,得到 {b}")
    if b == 1:
        return img.copy()  # 不做任何處理

    H, W = img.shape[:2]
    # 至少要能塞下一個 cell,否則 down size = 0
    new_w = max(1, W // b)
    new_h = max(1, H // b)

    # 步驟 1:下採樣(INTER_AREA 等同於對每個 cell 取平均)
    small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # 步驟 2:上採樣(INTER_NEAREST 讓每個 cell 變成 b×b 同色塊)
    out = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)
    return out


def pixelize_region(img: np.ndarray, bbox: BBox, b: int) -> np.ndarray:
    """對指定 bbox 區域做 pixelization,其他區域保持原樣。"""
    out = img.copy()
    x, y, w, h = bbox
    # clip 到圖內(防 bbox 超出)
    H, W = img.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return out  # bbox 無效就跳過
    region = img[y0:y1, x0:x1]
    out[y0:y1, x0:x1] = pixelize(region, b)
    return out


def pixelize_faces(
    img: np.ndarray,
    b: int,
    detector: FaceDetector | None = None,
    fallback_full: bool = True,
) -> tuple[np.ndarray, list[BBox]]:
    """
    偵測人臉 → 對每張臉的 bbox 區域 pixelize → 回傳結果與 bbox 列表。

    fallback_full=True:偵測不到臉就把整張圖當一個 bbox(ORL 用)。
    """
    if detector is None:
        detector = FaceDetector(backend="haar")
    boxes = detector.detect(img, fallback_full=fallback_full)
    out = img.copy()
    for box in boxes:
        out = pixelize_region(out, box, b)
    return out, boxes


# ---- CLI:批次處理整個資料夾 ------------------------------------------------
def _gather_images(root: Path) -> list[Path]:
    """遞迴蒐集 root 底下所有支援的影像檔。"""
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
    """處理整個資料夾,保留相對路徑結構。回傳處理張數。"""
    files = _gather_images(in_root)
    if not files:
        print(f"[pixelize] 警告:{in_root} 找不到任何影像", file=sys.stderr)
        return 0

    detector = FaceDetector(backend=backend) if detect_faces else None
    n = 0
    for f in files:
        rel = f.relative_to(in_root)
        # 寫成 png(無損)以避免 jpeg 二次壓縮影響後續評估
        out_path = out_root / rel.with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[pixelize] 跳過讀不到的檔:{f}", file=sys.stderr)
            continue

        if detect_faces:
            out, _ = pixelize_faces(img, b, detector=detector, fallback_full=True)
        else:
            out = pixelize(img, b)

        cv2.imwrite(str(out_path), out)
        n += 1
        if not quiet and n % 50 == 0:
            print(f"  處理 {n}/{len(files)} ...")
    return n


def main():
    p = argparse.ArgumentParser(description="批次對影像做 Pixelization 去識別化")
    p.add_argument("--input", "-i", required=True, help="輸入資料夾(會遞迴掃描)")
    p.add_argument("--output", "-o", required=True, help="輸出資料夾(保留相對路徑結構)")
    p.add_argument("--b", "-b", type=int, required=True, help="pixel block 邊長")
    p.add_argument(
        "--detect-faces",
        action="store_true",
        help="先偵測人臉再對 bbox 區域 pixelize(不加就對全圖)",
    )
    p.add_argument("--backend", choices=["haar", "hog"], default="haar")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    in_root = Path(args.input).resolve()
    out_root = Path(args.output).resolve()
    if not in_root.is_dir():
        p.error(f"input 不是資料夾:{in_root}")

    print(f"[pixelize] b={args.b}  detect_faces={args.detect_faces}")
    print(f"  in  : {in_root}")
    print(f"  out : {out_root}")
    n = _process_folder(in_root, out_root, args.b, args.detect_faces, args.backend, args.quiet)
    print(f"[pixelize] 完成,處理 {n} 張影像")


if __name__ == "__main__":
    main()
