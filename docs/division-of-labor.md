# 分工與交付清單

作業分三個 Step、五人分組。所有去識別化變體共用 `seed=42` 的 stratified train/test 切分（test_ratio=0.2），攻擊準確率才能彼此對照。

| 成員 | Step | 負責內容 | 交付檔案 | 狀態 |
|---|---|---|---|---|
| 成員 1 | 前置 + Step 1 | 資料集準備、人臉偵測、Pixelization（b=2,4,8,16） | `src/facedeid/dataset_loader.py`、`src/facedeid/face_detector.py`、`src/facedeid/pixelize.py`、`scripts/run_pixelize.sh`、`scripts/make_pixelize_comparison.py`、`figures/pixelize_comparison.png`、`data/README.md` | ✅ |
| 成員 2 | Step 1 + 整合 | Gaussian Blurring（k=15,45,99）、報告整合、本 repo 整理與最終打包 | `src/facedeid/gaussian_blur.py`、`scripts/run_blur.sh`、`scripts/make_blur_comparison.py`、`figures/blur_comparison.png`、`README.md`、最終報告（規劃 + 排版） | ✅（程式碼）／🟡（報告） |
| 成員 3 | Step 2 | CNN 架構與訓練 pipeline；對「每一種去識別化參數」獨立訓練（不混訓） | `src/facedeid/model.py`、`scripts/train.py`、`config.yaml`、訓練 log（loss/acc 曲線）、各參數 `.pth` | ⬜ |
| 成員 4 | Step 2 | CNN 評估（Top-1/Top-5）；跑完 8 組攻擊（原圖 + pix b∈{2,4,8,16} + blur k∈{15,45,99}）；整理對照表 | `scripts/evaluate.py`、完整攻擊結果表、攻擊分析文件 | ⬜ |
| 成員 5 | Step 3 | DP-Pixelization、DP-Blur；ε 掃描 {0.1,0.3,0.5,0.7,1,3,5}；MSE/SSIM；把 DP 影像交成員 4 重跑攻擊 | `src/facedeid/dp_pixelize.py`、`src/facedeid/dp_blur.py`、`scripts/compute_metrics.py`、MSE-vs-ε / SSIM-vs-ε 曲線、DP-vs-NP 對照表、敏感度推導文件 | ⬜ |

## 交付對接備註

- **資料集：** `data/att_faces/` 已是**真實 AT&T (ORL)**（400 張 92×112 PGM）；Step 1 的去識別化變體已產出（`outputs/pixelized/`、`outputs/blurred/`）。給其他成員的打包檔見 repo 上一層的 `HW3_Step1_deid_datasets.zip`（含 original + pix_b{2,4,8,16} + blur_k{15,45,99} + `split_{train,test}.json` + `dataset_loader.py`）。
- **成員 3、4：** 訓練資料直接用 `outputs/pixelized/pix_b{2,4,8,16}/` 與 `outputs/blurred/blur_k{15,45,99}/`，結構與 ORL 一致（`<class>/<image>.png`），用 `facedeid.DatasetIndex.from_att(...)` 載入；切分一律 `seed=42`（各變體檔名順序相同，所以同 seed 切出來會落在同一批影像上，已驗證一致）。
- **成員 5：** 可 `from facedeid import pixelize, gaussian_blur` 當作 NP（無噪音）baseline，在 cell 平均 / blur 後像素上加 Laplace 噪聲；`face_detector` 共用以確保 NP / DP 用同一個 bbox。敏感度推導需在報告 Step 3 清楚交代（DP-Pix：單一像素變動最多影響它所在 cell 平均 ±255/b²）。
- **報告（成員 2 統整）：** 三個 Step 的方法、實驗結果、討論、分工表。Step 2 / Step 3 的數字由成員 3/4/5 提供後回填。報告由成員 2 規劃內容、自行排版製作文件（不用自動生成腳本）。

## 待補進 repo 的東西

- `src/facedeid/model.py`、`scripts/train.py`、`config.yaml`、`scripts/evaluate.py`（成員 3、4）
- `src/facedeid/dp_pixelize.py`、`src/facedeid/dp_blur.py`、`scripts/compute_metrics.py`（成員 5）
- `figures/mse_vs_eps.png`、`figures/ssim_vs_eps.png`、攻擊準確率對照表（成員 4、5）
- `tests/`（pipeline 的最小單元測試）
- 最終報告檔（成員 2）
