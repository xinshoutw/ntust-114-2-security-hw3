"""
make_pixelize_comparison.py
---------------------------
產生作業 PDF 第 6 頁格式的視覺比較圖(Pixelization 版):
    [orig | b=2 | b=4 | b=8 | b=16]

每個 row 是同一個人(預設 4 row)。輸出到 figures/。

用法(在專案根目錄執行):
    uv run python scripts/make_pixelize_comparison.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from facedeid.pixelize import pixelize  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "att_faces"
OUT_DIR = PROJECT_ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)

B_VALUES = [2, 4, 8, 16]
SAMPLE_SUBJECTS = ["s1", "s5", "s10", "s20"]
SAMPLE_INDEX = 1
IMG_EXTS = {".pgm", ".png", ".jpg", ".jpeg", ".bmp"}


def _load_subject_image(subj: str):
    """讀 subj/SAMPLE_INDEX.pgm,讀不到就退而用該資料夾第一張圖。"""
    cand = DATA_ROOT / subj / f"{SAMPLE_INDEX}.pgm"
    if cand.exists():
        return cv2.imread(str(cand), cv2.IMREAD_GRAYSCALE)
    imgs = sorted(p for p in (DATA_ROOT / subj).glob("*") if p.suffix.lower() in IMG_EXTS)
    if not imgs:
        raise FileNotFoundError(f"{DATA_ROOT / subj} 底下找不到任何影像")
    return cv2.imread(str(imgs[0]), cv2.IMREAD_GRAYSCALE)


def main() -> None:
    n_rows, n_cols = len(SAMPLE_SUBJECTS), 1 + len(B_VALUES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 2.4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    col_titles = ["(a) orig"] + [f"({chr(ord('b') + i)}) b={b}" for i, b in enumerate(B_VALUES)]

    for r, subj in enumerate(SAMPLE_SUBJECTS):
        orig = _load_subject_image(subj)
        axes[r, 0].imshow(orig, cmap="gray", vmin=0, vmax=255)
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        if r == 0:
            axes[r, 0].set_title(col_titles[0])
        axes[r, 0].set_ylabel(subj, fontsize=10, rotation=0, labelpad=20, va="center")
        for c, b in enumerate(B_VALUES, start=1):
            axes[r, c].imshow(pixelize(orig, b), cmap="gray", vmin=0, vmax=255)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(col_titles[c])

    fig.suptitle("Pixelization: visual quality vs. block size b", y=1.00, fontsize=12)
    plt.tight_layout()
    out_png = OUT_DIR / "pixelize_comparison.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"視覺比較圖寫到:{out_png}")

    # 單列版本
    fig2, ax2 = plt.subplots(1, n_cols, figsize=(2.0 * n_cols, 2.4))
    orig = _load_subject_image(SAMPLE_SUBJECTS[0])
    ax2[0].imshow(orig, cmap="gray", vmin=0, vmax=255)
    ax2[0].set_xticks([]); ax2[0].set_yticks([]); ax2[0].set_title("(a) orig")
    for c, b in enumerate(B_VALUES, start=1):
        ax2[c].imshow(pixelize(orig, b), cmap="gray", vmin=0, vmax=255)
        ax2[c].set_xticks([]); ax2[c].set_yticks([])
        ax2[c].set_title(f"({chr(ord('b') + c - 1)}) b={b}")
    plt.tight_layout()
    out_png2 = OUT_DIR / "pixelize_comparison_single_row.png"
    plt.savefig(out_png2, dpi=150, bbox_inches="tight")
    print(f"單列版本寫到:{out_png2}")


if __name__ == "__main__":
    main()
