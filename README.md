# Face De-identification HW3 — 人臉去識別化方法與 AI 重識別攻擊

> 重現並延伸 Fan (2018, *Image Pixelization with Differential Privacy*) 的研究：在 AT&T (ORL) 人臉資料庫上實作兩種傳統去識別化方法、用 CNN 量測它們抵抗重識別攻擊的能力，再以 ε-差分隱私強化並比較效用與隱私的取捨。

---

## 總覽

「人眼看不出」不等於「機器認不出」。傳統的 **Pixelization**（以 `b×b` 區塊取平均）與 **Gaussian Blurring**（以 `k×k` 高斯核卷積）能讓人臉在肉眼下難以辨識，但卷積神經網路常常能從殘留的低頻結構中還原身分。本專案分三步驗證這件事：

- **Step 1 — 去識別化**：對 ORL 人臉套用 Pixelization（`b = 2, 4, 8, 16`）與 Gaussian Blurring（`k = 15, 45, 99`），輸出各參數的去識別化資料集與視覺比較圖。
- **Step 2 — AI 重識別攻擊**：對「每一種去識別化參數」獨立訓練一個 CNN（不混訓，依論文做法），量測 Top-1 / Top-5 重識別準確率；隨機猜測的 Top-1 ≈ 1/40 = 2.5%。
- **Step 3 — 差分隱私**：實作 **DP-Pixelization** 與 **DP-Blur**，掃描 `ε ∈ {0.1, 0.3, 0.5, 0.7, 1, 3, 5}`，以 **MSE / SSIM** 衡量影像效用、並重跑 CNN 攻擊驗證差分隱私確實壓低了重識別準確率。

所有去識別化變體共用同一份 train/test 切分（`seed = 42`），攻擊準確率才能彼此對照。

---

## 示例圖

| Pixelization（`b = 2 → 16`） | Gaussian Blurring（`k = 15 → 99`） |
|:---:|:---:|
| ![pixelize comparison](figures/pixelize_comparison.png) | ![blur comparison](figures/blur_comparison.png) |

含背景影像的「先偵測人臉 → 只去識別化 bbox 區域」示意：

![detect + region pixelize](figures/detect_pixelize_demo.png)

> `data/att_faces/` 為**真實的 AT&T (ORL) 人臉資料庫**（400 張 92×112 灰階 PGM，40 人各 10 張；來源見 [`data/README.md`](data/README.md)）。

---

## 開發規劃

| Step | 內容 | 狀態 |
|---|---|---|
| Step 1 | Pixelization（b=2,4,8,16）、人臉偵測、資料載入 | ✅ 完成 |
| Step 1 | Gaussian Blurring（k=15,45,99） | ✅ 完成 |
| Step 1 | 視覺比較圖 | ✅ 完成 |
| Step 2 | CNN 架構 + 訓練 pipeline（train/val/test 6:2:2，best ckpt 由 val 挑） | ✅ 完成 |
| Step 2 | Top-1/Top-5 評估（held-out test）+ 8 組攻擊實驗 + 對照表 | ✅ 完成 |
| Step 3 | DP-Pixelization b∈{2,4,8,16} + 兩種 DP-Blur 機制（LP-Blur, DP-Blur-Split） | ✅ 完成 |
| Step 3 | MSE/SSIM vs ε 曲線、DP-vs-NP 攻擊準確率對照（42 組） | ✅ 完成 |
| 文件 | 最終報告（方法 + 結果 + 討論 + 分工表） | 🟡 規劃中（由成員 2 統整、排版） |

詳細分工見 [`docs/division-of-labor.md`](docs/division-of-labor.md)。

---

## 給組員（必讀）

所有資料、checkpoint、reports、figures、DP 影像都直接放在 repo，不需要再下載外部 zip。要快速接手 Step 2 / Step 3，只要看這幾個地方：

| 你是 | 直接拿 | 一定要看 |
|---|---|---|
| **成員 3、4（CNN 攻擊）** | `data/deid/pixelized/pix_b{2,4,8,16}/`、`data/deid/blurred/blur_k{15,45,99}/`、`data/splits/split_{train,val,test}.json`（6:2:2 切分）、`checkpoints/{original,pix_b*,blur_k*}.pth`（8 個 Step 2 模型） | `docs/division-of-labor.md` 的「交付對接備註」、`src/facedeid/dataset_loader.py`（用 `DatasetIndex.from_att(...)` + `stratified_split_3way(..., seed=42)`）。**8 組各自獨立訓練，best ckpt 由 val acc 挑，test 全程 held-out** |
| **成員 5（差分隱私）** | `data/dp/{dp_pix_b{2,4,8,16},lp_blur,dp_blur_split}/`（42 組 DP 影像）、`data/dp/metrics.csv`、`checkpoints/dp_*.pth`（42 個 DP 攻擊模型） | 本 repo 使用 `scripts/train_evaluate_dp.py` 跑 CNN attack，結果整理於 `docs/dp-attack-results.md` |
| 想了解全貌 | — | 本 README 的「總覽」「專案架構」、`docs/division-of-labor.md` |

> 跑法：`uv sync` 之後 `./scripts/run_pixelize.sh` / `./scripts/run_blur.sh` 可重建 `data/deid/`；`uv run pytest` 跑煙霧測試。新增的去識別化變體一律沿用 `seed=42` 的切分。

---

## 系統需求

| 項目 | 需求                                                                 |
|---|--------------------------------------------------------------------|
| Python | 3.13（見 `.python-version`；最低相容 3.11）                                |
| 套件管理 | [uv](https://docs.astral.sh/uv/)                                   |
| 核心依賴 | `opencv-python`、`numpy`、`matplotlib`（見 `pyproject.toml`）           |
| 可選 | `dlib`（HOG+SVM 人臉偵測，編譯需 cmake）、`torch` / `torchvision`（Step 2 CNN） |
| 作業系統 | macOS / Linux（`scripts/*.sh` 為 bash 腳本）                            |

---

## 開發環境建置

```bash
# 1. 安裝 uv（若尚未安裝）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 建立環境並安裝依賴（uv 會依 .python-version 取得 Python 3.13）
uv sync

# 含可選依賴：
uv sync --extra detect-hog          # 加 dlib HOG+SVM 人臉偵測
uv sync --extra attack              # 加 torch / torchvision（Step 2）
uv sync --group dev                 # 加 ruff / pytest 等開發工具

# 3. 驗證
uv run python -c "import facedeid; print('facedeid OK')"
uv run pytest                       # 去識別化函式的煙霧測試
uv run ruff check src scripts tests # 風格檢查
```

### 跑去識別化 pipeline

```bash
# 0. 準備資料集：確認真實 ORL 位於 data/att_faces/s1 ... s40（見 data/README.md）

# 1. 產生去識別化資料集（輸出到 data/deid/）
./scripts/run_pixelize.sh           # → data/deid/pixelized/pix_b{2,4,8,16}
./scripts/run_blur.sh               # → data/deid/blurred/blur_k{15,45,99}

# 2. 產生視覺比較圖（輸出到 figures/）
uv run python scripts/make_pixelize_comparison.py
uv run python scripts/make_blur_comparison.py
uv run python scripts/make_detect_pixelize_demo.py

# 也可以直接呼叫模組 CLI：
uv run python -m facedeid.gaussian_blur --input data/att_faces --output data/deid/blurred/blur_k45 --k 45
uv run python -m facedeid.pixelize     --input data/att_faces --output data/deid/pixelized/pix_b8  --b 8
```

### 跑 CNN 重識別攻擊 pipeline

```bash
# 先安裝 Step 2 需要的 torch / torchvision
uv sync --extra attack

# 單獨訓練一組資料；auto 會優先使用 CUDA，其次 Apple MPS，最後 CPU
uv run --extra attack python scripts/train.py --dataset-root data/deid/pixelized/pix_b8 --name pix_b8 --config config.yaml --device auto

# 一次分別訓練 original + pix_b{2,4,8,16} + blur_k{15,45,99}
uv run --extra attack python scripts/train_all.py --config config.yaml --device auto

# Apple Silicon / M4 Pro 可明確指定 Metal GPU
uv run --extra attack python scripts/train_all.py --config config.yaml --device mps

# NVIDIA CUDA 可明確指定 CUDA GPU
uv run --extra attack python scripts/train_all.py --config config.yaml --device cuda

# 產生 loss / accuracy 曲線與 Top-1 / Top-5 summary
uv run --extra attack python scripts/plot_log.py --log-dir logs
uv run --extra attack python scripts/summarize_logs.py --log-dir logs --output reports/summary.csv

# 評估 8 組 checkpoint 的 Top-1 / Top-5 attack accuracy
uv run --extra attack python scripts/evaluate.py --all --device auto --output reports/evaluation.csv

# DP 影像資料集已直接放在 data/dp/，可訓練並評估 DP 版本的 attack accuracy
uv run --extra attack python scripts/train_evaluate_dp.py --device auto --output reports/dp_evaluation.csv
```

### 在程式碼中使用

```python
from facedeid import DatasetIndex, stratified_split, load_image
from facedeid import pixelize, gaussian_blur, FaceDetector

idx = DatasetIndex.from_att("data/att_faces")
train, test = stratified_split(idx, test_ratio=0.2, seed=42)   # 全專案統一 seed=42

img = load_image(train.items[0][0])      # numpy uint8（預設灰階）
p_out = pixelize(img, b=8)               # Pixelization
g_out = gaussian_blur(img, k=45)         # Gaussian Blurring（sigma=0 → 由 k 自動推算）

# 含背景影像：偵測人臉 + 只處理 bbox 區域
det = FaceDetector(backend="haar")       # 或 backend="hog"（需 dlib）
g_out, boxes = facedeid.gaussian_blur_faces(img, k=45, detector=det, fallback_full=True)
```

---

## 專案架構

```
face-deid-hw3/
├── README.md
├── pyproject.toml                  # uv 專案定義與依賴
├── .python-version                 # 3.13
├── .gitignore
├── data/
│   ├── README.md                   # ORL / FaceScrub / CelebA 下載與目錄結構說明
│   ├── att_faces/                  # AT&T (ORL)：s1/1.pgm ... s40/10.pgm（真實資料）+ ORL_README.txt
│   ├── deid/                       # Step 1 產出：傳統去識別化影像
│   │   ├── pixelized/pix_b{2,4,8,16}/  #   Pixelization 各 b（PNG，結構同 ORL）
│   │   └── blurred/blur_k{15,45,99}/    #   Gaussian Blur 各 k
│   ├── dp/                         # Step 3 產出：DP 影像 + metrics
│   │   ├── dp_pix_b8/eps{0.1..5.0}/     #   DP-Pix b=8
│   │   ├── dp_pix_b16/eps{0.1..5.0}/    #   DP-Pix b=16
│   │   ├── dp_blur/eps{0.1..5.0}/       #   DP-Blur
│   │   └── metrics.csv             #   MSE / SSIM
│   └── splits/                     # 統一 train/test 切分（seed=42）
│       ├── split_train.json        #   320 張
│       └── split_test.json         #   80 張
├── checkpoints/                    # 8 個 Step 2 + 21 個 Step 3 DP 攻擊模型
├── reports/                        # evaluation.csv、per_class/、hardest_classes.csv
├── src/
│   └── facedeid/                   # 主套件
│       ├── __init__.py             # 對外 re-export
│       ├── dataset_loader.py       # 統一資料載入 / stratified train-test split（seed=42）
│       ├── face_detector.py        # 人臉偵測（Haar Cascade / dlib HOG+SVM）
│       ├── pixelize.py             # Pixelization（b×b 區塊取平均；含 CLI）
│       └── gaussian_blur.py        # Gaussian Blurring（k×k 高斯核卷積；含 CLI）
├── scripts/
│   ├── run_pixelize.sh             # 批次 b=2,4,8,16
│   ├── run_blur.sh                 # 批次 k=15,45,99
│   ├── make_pixelize_comparison.py # Pixelization 視覺比較圖
│   ├── make_blur_comparison.py     # Gaussian Blur 視覺比較圖
│   ├── make_detect_pixelize_demo.py# 偵測 + 區域去識別化示意圖
│   ├── evaluate.py                 # Step 2 Top-1 / Top-5 評估
│   └── train_evaluate_dp.py        # Step 3 DP 資料的 CNN attack 驗證
├── tests/
│   └── test_smoke.py               # 去識別化函式的最小煙霧測試（uv run pytest）
├── figures/                        # 報告用視覺比較圖 + per_class_top1 熱圖
└── docs/
    └── division-of-labor.md        # 五人分工與交付清單
```

> `data/deid/` 與 `data/dp/` 已直接 commit 進 repo，不依賴外部 zip。
> `data/deid/` 可由 `./scripts/run_*.sh` 從 `data/att_faces/` 重建；`data/dp/` 由成員 5 的 `scripts/dp_*.py` 產出。

> Step 2 的 CNN 訓練與評估 pipeline 在 `src/facedeid/model.py`、`scripts/train.py`、`scripts/train_all.py`、`scripts/plot_log.py`、`scripts/summarize_logs.py`、`scripts/evaluate.py` 與 `config.yaml`。Step 3 的 DP 影像由 `scripts/dp_pixelization.py` / `scripts/dp_blur.py` 產出，CNN attack 驗證走 `scripts/train_evaluate_dp.py`。

---

## Caveats（須在報告中誠實揭露）

### N=80 的統計侷限

ORL 一共 400 張影像、每 class 10 張，切分 6:2:2 後 test set 只剩 80 張（每 class 2 張）。**單一預測誤差 = 1.25 個百分點。** 因此：

- 跨 dataset 之間 5–10pp 的 Top-1 差異還在 4–8 筆預測之內，方向性可解讀但**不適合宣告嚴格單調趨勢**。
- 可靠的結論：「在所有去識別化參數下 CNN Top-1 都遠優於 1/40 隨機猜測」。
- LP-Blur ε=0.5 的 Top-1=0.5375 是這個樣本量下的明顯 outlier；其他 6 個 ε 都在 0.84–0.89。MPS 上 PyTorch 的 non-determinism 讓跨 run 差個 5–10pp 是正常範圍。

### DP-Blur-Split 的 sensitivity bound 比理論最緊更保守

實作用 `Laplace(scale = 255·k²/ε)`，相當於把單一 output pixel 的 sensitivity 假設為 255。**理論最緊的 bound 是 max kernel weight × 255**，對 Gaussian kernel 約等於 `0.18·255 ≈ 46`。我們選用較大的 scale 是為了**保證嚴格 ε-DP**——更多的雜訊不會破壞 DP 保證、只會犧牲更多 utility。報告中可說「我們以可證明的保守上界實作；緊上界版本會給出較好的 utility 但需要更小心的證明」。

### DP-Pix 用的是 cell-mean 機制（parallel composition）

不是 Fan 2018 完整版的 m-neighborhood mechanism。對 ORL 已 cropped 的人臉而言，這個簡化版的 sensitivity `255/b²` 是 m=1 時的特例，足以呈現 privacy-utility trade-off 並比較 b 值的影響。

---

## 貢獻

本專案為五人分組作業，分工如下（完整交付清單見 [`docs/division-of-labor.md`](docs/division-of-labor.md)）：

1. **成員 1** — 資料集前置、人臉偵測、Pixelization（b=2,4,8,16）：`dataset_loader.py`、`face_detector.py`、`pixelize.py`、`run_pixelize.sh`、視覺比較圖、資料集說明。
2. **成員 2** — Gaussian Blurring（k=15,45,99）、報告整合、最終打包：`gaussian_blur.py`、`run_blur.sh`、視覺比較圖、最終報告（規劃/排版）、本 repo 整理。
3. **成員 3** — CNN 架構與訓練 pipeline（train/val/test 三切、每組去識別化參數獨立訓練）：`model.py`、`train.py`、`config.yaml`、訓練 log、各參數 `.pth`。
4. **成員 4** — CNN 評估與攻擊實驗：`evaluate.py`（Top-1/Top-5）、Step 2 八組 + Step 3 四十二組攻擊結果對照表、攻擊分析。
5. **成員 5** — 差分隱私：DP 影像資料集（DP-Pix b∈{2,4,8,16}、LP-Blur、DP-Blur-Split）、`metrics.csv`、MSE/SSIM 曲線、敏感度推導文件；本 repo 包含完整 DP-vs-NP CNN attack 對照表。

提交前請確認：

1. `uv run ruff check src scripts` 無錯誤（已安裝 dev group 時）。
2. 新增模組在 `src/facedeid/__init__.py` 有對應 re-export（如適用）。
3. 任何新的去識別化變體都沿用 `seed=42` 的 train/test 切分。
4. 不要把 `.venv/`、`logs/`、`plots/`、`__pycache__/` 等 transient 產物 commit 進 repo。

---

## 授權與致謝

本專案為課程作業，僅供教學與研究使用。

- AT&T (ORL) Database of Faces — AT&T Laboratories Cambridge，<https://cam-orl.co.uk/facedatabase.html>
- L. Fan, *Image Pixelization with Differential Privacy*, DBSec 2018（亦見 TPDP 2019）
- C. Dwork and A. Roth, *The Algorithmic Foundations of Differential Privacy*, 2014
