"""
make_detect_pixelize_demo.py
----------------------------
展示「先偵測人臉 → 只對 bbox 區域做去識別化」的 pipeline,適用於 FaceScrub /
CelebA / 其他含背景的影像。由於沒有 FaceScrub 樣本,這裡合成一張「人臉貼在
背景上」的測試圖(實際使用時把 make_synthetic_scene() 換成讀 FaceScrub 任一張即可)。

輸出到 figures/detect_pixelize_demo.png。

用法(在專案根目錄執行):
    uv run python scripts/make_detect_pixelize_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from facedeid.face_detector import FaceDetector  # noqa: E402
from facedeid.pixelize import pixelize_faces  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)


def make_synthetic_scene() -> np.ndarray:
    """合成一張 256×256 場景:有紋理背景,中央貼一張放大的 synthetic ORL 人臉。"""
    rng = np.random.RandomState(0)
    H = W = 256
    yy, xx = np.mgrid[0:H, 0:W]
    bg = (((xx + yy) // 4) % 200 + 30).astype(np.uint8)
    bg = np.clip(bg.astype(np.int16) + rng.randint(-15, 15, (H, W)), 0, 255).astype(np.uint8)

    face_path = PROJECT_ROOT / "data" / "att_faces" / "s1" / "1.pgm"
    face = cv2.imread(str(face_path), cv2.IMREAD_GRAYSCALE)
    if face is None:
        raise FileNotFoundError(f"找不到 {face_path}(先跑 scripts/make_synthetic_orl.py 或放好真實 ORL)")
    face_big = cv2.resize(face, (120, 146), interpolation=cv2.INTER_LINEAR)
    fh, fw = face_big.shape
    cy, cx = (H - fh) // 2, (W - fw) // 2
    bg[cy:cy + fh, cx:cx + fw] = face_big
    return bg


def main() -> None:
    img = make_synthetic_scene()
    det = FaceDetector(backend="haar")
    boxes = det.detect(img, fallback_full=False)
    print(f"偵測到 {len(boxes)} 張臉:{boxes}")

    b_values = [4, 8, 16]
    fig, axes = plt.subplots(1, 2 + len(b_values), figsize=(3 * (2 + len(b_values)), 3))

    axes[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("(a) original"); axes[0].set_xticks([]); axes[0].set_yticks([])

    drawn = det.draw(img, boxes)
    axes[1].imshow(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB))
    axes[1].set_title("(b) detected"); axes[1].set_xticks([]); axes[1].set_yticks([])

    for i, b in enumerate(b_values, start=2):
        out, _ = pixelize_faces(img, b, detector=det, fallback_full=False)
        axes[i].imshow(out, cmap="gray", vmin=0, vmax=255)
        axes[i].set_title(f"({chr(ord('a') + i)}) b={b} (face only)")
        axes[i].set_xticks([]); axes[i].set_yticks([])

    plt.tight_layout()
    out_png = OUT_DIR / "detect_pixelize_demo.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"demo 寫到:{out_png}")


if __name__ == "__main__":
    main()
