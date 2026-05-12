"""
make_synthetic_orl.py
---------------------
產生 synthetic ORL-like 資料集,用來在沒有真實 ORL 下載權限時驗證整條 pipeline。
模擬 ORL 結構:40 個 subject × 10 張 92×112 灰階 PGM,寫到 data/att_faces/。
每個 subject 用一張隨機 base 臉,加上 10 種不同的 affine + 亮度擾動。

⚠️ 正式實驗請換成真實 ORL(合成資料的 subject 間區別性過低,CNN 攻擊準確率會偏離論文結果)。

用法(在專案根目錄執行):
    uv run python scripts/make_synthetic_orl.py
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

np.random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT = PROJECT_ROOT / "data" / "att_faces"
ROOT.mkdir(parents=True, exist_ok=True)

W, H = 92, 112
N_SUBJ, N_IMG = 40, 10


def make_base_face(seed: int) -> np.ndarray:
    """製作一張類似人臉結構的灰階 base 圖。"""
    rng = np.random.RandomState(seed)
    img = np.full((H, W), rng.randint(40, 90), dtype=np.uint8)
    cx, cy = W // 2, H // 2 + 5
    axes = (rng.randint(28, 34), rng.randint(40, 46))
    skin_val = rng.randint(140, 200)
    cv2.ellipse(img, (cx, cy), axes, 0, 0, 360, int(skin_val), -1)
    eye_y = cy - 12
    for ex_off in (-12, 12):
        cv2.circle(img, (cx + ex_off, eye_y), 4, int(rng.randint(20, 60)), -1)
        cv2.circle(img, (cx + ex_off, eye_y), 2, 255, -1)
    cv2.line(img, (cx, cy - 5), (cx, cy + 8), int(skin_val - 30), 2)
    mouth_y = cy + 18
    cv2.ellipse(img, (cx, mouth_y), (8, 3), 0, 0, 360, int(rng.randint(30, 80)), -1)
    for ex_off in (-12, 12):
        cv2.line(img, (cx + ex_off - 5, eye_y - 8), (cx + ex_off + 5, eye_y - 8),
                 int(rng.randint(10, 50)), 2)
    return img


def perturb(img: np.ndarray, j: int) -> np.ndarray:
    """產生一張同一 subject 的不同樣本(輕微旋轉、平移、亮度、雜訊)。"""
    rng = np.random.RandomState(j * 17 + 3)
    angle = rng.uniform(-8, 8)
    tx, ty = rng.uniform(-3, 3), rng.uniform(-3, 3)
    M = cv2.getRotationMatrix2D((W // 2, H // 2), angle, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    out = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
    out = np.clip(out.astype(np.int16) + rng.randint(-20, 20), 0, 255).astype(np.uint8)
    noise = rng.normal(0, 4, out.shape).astype(np.int16)
    return np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def write_pgm(path: Path, img: np.ndarray) -> None:
    """寫成 P5 binary PGM(跟真實 ORL 同格式)。"""
    with open(path, "wb") as f:
        f.write(f"P5\n{W} {H}\n255\n".encode())
        f.write(img.tobytes())


def main() -> None:
    for sid in range(1, N_SUBJ + 1):
        sdir = ROOT / f"s{sid}"
        sdir.mkdir(exist_ok=True)
        base = make_base_face(seed=sid * 7)
        for j in range(1, N_IMG + 1):
            write_pgm(sdir / f"{j}.pgm", perturb(base, j))
    print(f"生成完成:{N_SUBJ} subjects × {N_IMG} 張,共 {N_SUBJ * N_IMG} 張 → {ROOT}")


if __name__ == "__main__":
    main()
