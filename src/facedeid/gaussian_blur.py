"""
gaussian_blur.py
----------------
Gaussian Blurring 去識別化主程式(成員 2 交付)。

演算法(對應作業 PDF Step 1 的第二種去識別化方法):
    對人臉區域套用一個 k×k 的高斯核做卷積:
        I'(x, y) = Σ_{i,j} G(i, j) · I(x - i, y - j)
    其中 G 是 2D 高斯函式
        G(i, j) = 1 / (2πσ²) · exp( -(i² + j²) / (2σ²) )
    k 越大 → 模糊越強、視覺品質越差、去識別化程度越高。

關於 σ(sigma):
    OpenCV 的 cv2.GaussianBlur 若把 sigma 設成 0,會用核大小自動推得:
        σ = 0.3 · ((k - 1) · 0.5 - 1) + 0.8
    這是常見作法(Fan 2018、多數實作都這樣),所以本程式預設 sigma=0,
    讓 σ 隨 k 一起放大。需要固定 σ 時可用 --sigma 指定。

支援兩種使用方式(跟成員 1 的 pixelize.py 一致):
    * 對「全圖」做 blur:適合 ORL 這種已 cropped 的人臉
    * 對「人臉 bbox 區域」做 blur:適合 FaceScrub / 自然影像
      (用 face_detector 抓 bbox,只把 bbox 區域 blur 後貼回原圖)

用法:
    # 程式碼
    from facedeid.gaussian_blur import gaussian_blur, gaussian_blur_faces
    out = gaussian_blur(img, k=45)

    # CLI:批次處理整個資料夾(在專案根目錄用 uv run)
    uv run python -m facedeid.gaussian_blur --input data/att_faces --output outputs/blurred/blur_k45 --k 45
    uv run python -m facedeid.gaussian_blur --input data/celeb --output outputs/celeb_blur_k45 --k 45 --detect-faces
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from .face_detector import FaceDetector, BBox


# ---- 核心演算法 -------------------------------------------------------------
def _normalize_ksize(k: int) -> int:
    """高斯核大小必須是正奇數;傳入偶數就 +1,傳入 < 1 直接報錯。"""
    if k < 1:
        raise ValueError(f"k 必須 ≥ 1,得到 {k}")
    if k % 2 == 0:
        k += 1  # OpenCV 要求奇數核
    return k


def gaussian_blur(img: np.ndarray, k: int, sigma: float = 0.0) -> np.ndarray:
    """
    對整張影像套用 k×k 高斯模糊。

    參數:
      img  : numpy array,(H, W) 灰階 或 (H, W, 3) BGR
      k    : 高斯核邊長(會自動修正為正奇數)
      sigma: 高斯標準差;0 表示讓 OpenCV 依 k 自動推算(預設,常見作法)

    回傳: 與 img 同 shape 同 dtype 的模糊影像。
    """
    k = _normalize_ksize(k)
    if k == 1:
        return img.copy()  # 1×1 核 = 不做任何處理
    return cv2.GaussianBlur(img, (k, k), sigmaX=float(sigma), sigmaY=float(sigma))


def gaussian_blur_region(
    img: np.ndarray, bbox: BBox, k: int, sigma: float = 0.0
) -> np.ndarray:
    """對指定 bbox 區域做高斯模糊,其他區域保持原樣。"""
    out = img.copy()
    x, y, w, h = bbox
    H, W = img.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return out  # bbox 無效就跳過
    region = img[y0:y1, x0:x1]
    out[y0:y1, x0:x1] = gaussian_blur(region, k, sigma=sigma)
    return out


def gaussian_blur_faces(
    img: np.ndarray,
    k: int,
    detector: FaceDetector | None = None,
    fallback_full: bool = True,
    sigma: float = 0.0,
) -> tuple[np.ndarray, list[BBox]]:
    """
    偵測人臉 → 對每張臉的 bbox 區域 blur → 回傳結果與 bbox 列表。

    fallback_full=True:偵測不到臉就把整張圖當一個 bbox(ORL 用)。
    """
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
    """處理整個資料夾,保留相對路徑結構。回傳處理張數。"""
    files = _gather_images(in_root)
    if not files:
        print(f"[gaussian_blur] 警告:{in_root} 找不到任何影像", file=sys.stderr)
        return 0

    detector = FaceDetector(backend=backend) if detect_faces else None
    n = 0
    for f in files:
        rel = f.relative_to(in_root)
        # 寫成 png(無損),避免 jpeg 二次壓縮影響後續 CNN 攻擊評估
        out_path = out_root / rel.with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[gaussian_blur] 跳過讀不到的檔:{f}", file=sys.stderr)
            continue

        if detect_faces:
            out, _ = gaussian_blur_faces(
                img, k, detector=detector, fallback_full=True, sigma=sigma
            )
        else:
            out = gaussian_blur(img, k, sigma=sigma)

        cv2.imwrite(str(out_path), out)
        n += 1
        if not quiet and n % 50 == 0:
            print(f"  處理 {n}/{len(files)} ...")
    return n


def main():
    p = argparse.ArgumentParser(description="批次對影像做 Gaussian Blurring 去識別化")
    p.add_argument("--input", "-i", required=True, help="輸入資料夾(會遞迴掃描)")
    p.add_argument("--output", "-o", required=True, help="輸出資料夾(保留相對路徑結構)")
    p.add_argument("--k", "-k", type=int, required=True, help="高斯核邊長(偶數會自動 +1)")
    p.add_argument(
        "--sigma",
        type=float,
        default=0.0,
        help="高斯標準差;0(預設)表示讓 OpenCV 依 k 自動推算",
    )
    p.add_argument(
        "--detect-faces",
        action="store_true",
        help="先偵測人臉再對 bbox 區域 blur(不加就對全圖)",
    )
    p.add_argument("--backend", choices=["haar", "hog"], default="haar")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    in_root = Path(args.input).resolve()
    out_root = Path(args.output).resolve()
    if not in_root.is_dir():
        p.error(f"input 不是資料夾:{in_root}")

    eff_k = _normalize_ksize(args.k)
    print(f"[gaussian_blur] k={eff_k}  sigma={args.sigma or 'auto'}  detect_faces={args.detect_faces}")
    print(f"  in  : {in_root}")
    print(f"  out : {out_root}")
    n = _process_folder(
        in_root, out_root, args.k, args.sigma, args.detect_faces, args.backend, args.quiet
    )
    print(f"[gaussian_blur] 完成,處理 {n} 張影像")


if __name__ == "__main__":
    main()
