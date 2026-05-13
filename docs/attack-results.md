# Step 2 CNN Attack Results

本頁整理 CNN 對原圖與去識別化影像的重識別攻擊結果。所有資料集共用 `outputs/split_train.json` 與 `outputs/split_test.json`，每一種去識別化參數都獨立訓練與評估一個 CNN，沒有混合訓練。

完整的「為什麼」分析另放於 [`attack-analysis.md`](attack-analysis.md)：per-dataset 解讀、pix 為什麼不單調、blur 為什麼單調、N=80 統計侷限、為什麼仍需要 Step 3 DP。

訓練曲線（loss / accuracy vs epoch）見 [`../plots/`](../plots/)，每組 best epoch 與 train/test loss 摘要見 [`../reports/summary.csv`](../reports/summary.csv)。

執行重現：

```bash
uv run --extra attack python scripts/train_all.py --config config.yaml --device auto         # 訓練 8 組
uv run --extra attack python scripts/evaluate.py --all --device auto --output reports/evaluation.csv  # 評估
uv run --extra attack python scripts/plot_log.py --log-dir logs --output-dir plots           # 訓練曲線
uv run --extra attack python scripts/summarize_logs.py --log-dir logs --output reports/summary.csv    # summary
```

## Top-1 / Top-5 Accuracy

下表是以 `seed=42`、100 epoch 重跑 `scripts/train_all.py` 後，再以 `scripts/evaluate.py --all` 量測 8 組 best-epoch checkpoint 的結果（committed checkpoints 由 `chore(checkpoints)` 提供，原始輸出於 [`../reports/evaluation.csv`](../reports/evaluation.csv)，可直接重現）。

| Dataset | Top-1 Accuracy | Top-5 Accuracy | Test Loss | Test Samples | Best Epoch |
|---|---:|---:|---:|---:|---:|
| original | 0.9000 | 0.9875 | 0.4385 | 80 | 93 |
| pix_b2 | 0.9250 | 0.9750 | 0.4187 | 80 | 73 |
| pix_b4 | 0.9000 | 0.9875 | 0.4669 | 80 | 95 |
| pix_b8 | 0.9125 | 0.9875 | 0.4359 | 80 | 95 |
| pix_b16 | 0.9000 | 1.0000 | 0.4899 | 80 | 97 |
| blur_k15 | 0.9125 | 0.9875 | 0.3857 | 80 | 93 |
| blur_k45 | 0.8875 | 0.9875 | 0.5596 | 80 | 99 |
| blur_k99 | 0.8625 | 0.9875 | 0.5859 | 80 | 96 |

> PyTorch 在 MPS / CPU 上不是完全 deterministic，數字會在 ±3pp（≤ 3 筆預測）內抖動。本表為單一 run（`seed=42`）的結果；趨勢結論在多 run 下仍成立。

## Analysis（短版）

隨機猜測 40 類的 Top-1 ≈ 1/40 = 2.5%、Top-5 ≈ 12.5%。實驗結果顯示，即使經過 Pixelization 或 Gaussian Blur，CNN 仍能取得 **86.25% 至 92.50%** 的 Top-1（**至少 34× 隨機**），代表傳統去識別化基本無法阻止 AI 重識別攻擊。

Gaussian Blur 隨 k 變大整體趨於下降，最強的 `blur_k99` Top-1 = 86.25%、Test Loss 0.59，是 8 組中最低的攻擊準確率；模型對強模糊影像的信心顯著降低。

Pixelization 在 ORL 上**沒有呈現嚴格單調下降**——所有 4 組 Top-1 都落在 90.00-92.50% 區間。這指向 ORL 的識別線索集中在低頻訊號（整體輪廓、髮量、頭顱形狀），而 pixelization 本質上就是低通濾波，保留了大部分低頻成分。

整體最低 Top-1 仍維持在 86% 以上，距離隨機猜測差兩個數量級，**這正是 Step 3 引入 ε-DP 的動機**。

> 完整的「為什麼」分析（per-dataset 解讀、pix 不單調原因、blur 單調原因、N=80 統計侷限、為什麼仍需要 Step 3 DP、訓練曲線觀察、最終報告建議）見 [`attack-analysis.md`](attack-analysis.md)。

## 訓練曲線

每組 100 epoch 的 train/test loss 與 Top-1/Top-5 accuracy 曲線見：

```
plots/original_{loss,accuracy}.png
plots/pix_b{2,4,8,16}_{loss,accuracy}.png
plots/blur_k{15,45,99}_{loss,accuracy}.png
```

共 16 張 PNG，由 `scripts/plot_log.py --log-dir logs --output-dir plots` 產出。

## 每組 Best Epoch 摘要

完整摘要在 [`../reports/summary.csv`](../reports/summary.csv)，best epoch 集中在 73–99，全部接近 100 epoch 上限，代表模型仍在改進；增加 epoch 可能略提天花板，但**不會改變相對排序**。
