import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from skimage.metrics import structural_similarity, mean_squared_error

from dp_pixelization import dp_pixelization
from dp_blur import dp_blur

EPSILONS    = [0.1, 0.3, 0.5, 0.7, 1.0, 3.0, 5.0]
BLOCK_SIZES = [8, 16]
SRC_ROOT    = "data/att_faces"
OUT_ROOT    = "output"


# ── 1. 載入影像 ────────────────────────────────────────────────────────────────

def load_images(src_root: str):
    src = Path(src_root)
    imgs = []
    for pgm in sorted(src.rglob("*.pgm")):
        img = cv2.imread(str(pgm), cv2.IMREAD_GRAYSCALE)
        imgs.append((pgm, img))
    return imgs


def calc_metrics(original: np.ndarray, processed: np.ndarray):
    mse  = mean_squared_error(original, processed)
    ssim = structural_similarity(original, processed, data_range=255)
    return mse, ssim


# ── 2. ε 掃描 + 存 CSV ─────────────────────────────────────────────────────────

def run_scan():
    images = load_images(SRC_ROOT)
    rows = []

    for eps in EPSILONS:
        # DP-Pixelization
        for b in BLOCK_SIZES:
            mse_list, ssim_list = [], []
            dst = Path(OUT_ROOT) / f"dp_pix_b{b}" / f"eps{eps}"
            dst.mkdir(parents=True, exist_ok=True)

            for pgm, img in images:
                result = dp_pixelization(img, b, eps)
                rel = pgm.relative_to(SRC_ROOT)
                out_path = dst / rel.with_suffix(".png")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path), result)

                mse, ssim = calc_metrics(img, result)
                mse_list.append(mse)
                ssim_list.append(ssim)

            rows.append({
                "method":  f"DP-Pix-b{b}",
                "epsilon": eps,
                "MSE":     float(np.mean(mse_list)),
                "SSIM":    float(np.mean(ssim_list)),
            })
            print(f"DP-Pix b={b}  ε={eps}  MSE={rows[-1]['MSE']:.2f}  SSIM={rows[-1]['SSIM']:.4f}")

        # DP-Blur
        mse_list, ssim_list = [], []
        dst = Path(OUT_ROOT) / "dp_blur" / f"eps{eps}"
        dst.mkdir(parents=True, exist_ok=True)

        for pgm, img in images:
            result = dp_blur(img, eps)
            rel = pgm.relative_to(SRC_ROOT)
            out_path = dst / rel.with_suffix(".png")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), result)

            mse, ssim = calc_metrics(img, result)
            mse_list.append(mse)
            ssim_list.append(ssim)

        rows.append({
            "method":  "DP-Blur",
            "epsilon": eps,
            "MSE":     float(np.mean(mse_list)),
            "SSIM":    float(np.mean(ssim_list)),
        })
        print(f"DP-Blur      ε={eps}  MSE={rows[-1]['MSE']:.2f}  SSIM={rows[-1]['SSIM']:.4f}")

    csv_path = Path(OUT_ROOT) / "metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "epsilon", "MSE", "SSIM"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nmetrics saved → {csv_path}")
    return rows


# ── 3. 畫曲線圖 ────────────────────────────────────────────────────────────────

def plot_curves(rows: list):
    # 整理成 {method: {eps: [], MSE: [], SSIM: []}}
    data = defaultdict(lambda: {"eps": [], "MSE": [], "SSIM": []})
    for row in rows:
        m = row["method"]
        data[m]["eps"].append(row["epsilon"])
        data[m]["MSE"].append(row["MSE"])
        data[m]["SSIM"].append(row["SSIM"])

    styles = {
        "DP-Pix-b8":  {"marker": "o", "linestyle": "-",  "color": "#2196F3"},
        "DP-Pix-b16": {"marker": "s", "linestyle": "--", "color": "#4CAF50"},
        "DP-Blur":    {"marker": "^", "linestyle": "-.", "color": "#F44336"},
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for method, d in data.items():
        kw = styles.get(method, {"marker": "x", "linestyle": ":"})
        ax1.plot(d["eps"], d["MSE"],  label=method, **kw)
        ax2.plot(d["eps"], d["SSIM"], label=method, **kw)

    # MSE 圖
    ax1.set_xlabel("Privacy Budget ε", fontsize=12)
    ax1.set_ylabel("MSE", fontsize=12)
    ax1.set_title("MSE vs ε", fontsize=13, fontweight="bold")
    ax1.set_xscale("log")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # SSIM 圖
    ax2.set_xlabel("Privacy Budget ε", fontsize=12)
    ax2.set_ylabel("SSIM", fontsize=12)
    ax2.set_title("SSIM vs ε", fontsize=13, fontweight="bold")
    ax2.set_xscale("log")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(OUT_ROOT) / "curves.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"curves saved → {out_path}")


# ── 4. 主程式 ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rows = run_scan()
    plot_curves(rows)