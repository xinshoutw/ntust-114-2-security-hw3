# Step 2 CNN Attack Results

本頁整理 CNN 對原圖與去識別化影像的重識別攻擊結果。所有資料集共用 `data/splits/split_{train,val,test}.json`，每一種去識別化參數都獨立訓練與評估一個 CNN，沒有混合訓練。

完整的「為什麼」分析另放於 [`attack-analysis.md`](attack-analysis.md)：per-dataset 解讀、pix 為什麼不單調、blur 為什麼單調、N=80 統計侷限、為什麼仍需要 Step 3 DP。

訓練曲線（loss / accuracy vs epoch）見 [`../plots/`](../plots/)，每組 best epoch 與 train/val/test loss 摘要見 [`../reports/summary.csv`](../reports/summary.csv)。

執行重現：

```bash
uv run --extra attack python scripts/train_all.py --config config.yaml --device auto         # 訓練 8 組
uv run --extra attack python scripts/evaluate.py --all --device auto --output reports/evaluation.csv  # 評估
uv run --extra attack python scripts/plot_log.py --log-dir logs --output-dir plots           # 訓練曲線
uv run --extra attack python scripts/summarize_logs.py --log-dir logs --output reports/summary.csv    # summary
```

## 方法學

- 切分：每 class 10 張 → 6 train / 2 val / 2 test（`stratified_split_3way`, `seed=42`）。共 240 train / 80 val / 80 test。
- Best checkpoint 以 **val accuracy** 挑選，**test set 全程 held-out**。`evaluate.py` 報告的 Top-1/Top-5 是這個 held-out test set 的數字。
- 模型：`SimpleCNN`（3 conv blocks + classifier，輸入 128×128），SGD lr=0.01 momentum=0.9，batch=128，100 epoch。

## Top-1 / Top-5 Accuracy（held-out test set, N=80）

來源：[`../reports/evaluation.csv`](../reports/evaluation.csv)。

| Dataset | Top-1 Accuracy | Top-5 Accuracy | Test Loss | Best Epoch (val) |
|---|---:|---:|---:|---:|
| original | 0.8500 | 0.9750 | 0.5542 | 92 |
| pix_b2 | 0.8500 | 0.9875 | 0.4455 | 90 |
| pix_b4 | 0.8500 | 0.9875 | 0.5528 | 100 |
| pix_b8 | 0.8250 | 1.0000 | 0.6559 | 96 |
| pix_b16 | 0.7625 | 0.9750 | 0.8002 | 97 |
| blur_k15 | 0.8625 | 0.9750 | 0.6050 | 94 |
| blur_k45 | 0.8125 | 0.9750 | 0.7022 | 97 |
| blur_k99 | 0.8125 | 0.9750 | 0.8301 | 100 |

> PyTorch 在 MPS / CPU 上不是完全 deterministic，跨 run 數字會在 ±3pp 內抖動（test set 1 筆預測 = 1.25pp）。本表為單一 run（`seed=42`）。

## Analysis（短版）

隨機猜測 40 類的 Top-1 ≈ 1/40 = 2.5%、Top-5 ≈ 12.5%。實驗顯示即使經過 Pixelization 或 Gaussian Blur，CNN 仍能取得 **76.25% 至 86.25%** 的 Top-1（**至少 30× 隨機**），傳統去識別化擋不住 CNN 攻擊。

兩條觀察值得寫進報告：

1. **Pixelization 在 ORL 上開始呈現可見下降**——`pix_b16` (76.25%) 比 `original` (85%) 低 8.75pp，比舊版（test-set peeking 下不單調）有更清楚的訊號。`pix_b2/b4` 仍與 original 持平，`pix_b8` (82.5%) 介於中間。
2. **Gaussian Blur 隨 k 變大整體下降**：blur_k15 (86.25%) → blur_k99 (81.25%)，總共 5pp 下降但 Top-5 仍 97.5%，**模糊不會把人臉變成「另一個人」**。

整體最低 Top-1（pix_b16 76.25%）仍是隨機的 30 倍以上，**這就是 Step 3 引入 ε-DP 的動機**——傳統去識別化不夠。

> 完整討論見 [`attack-analysis.md`](attack-analysis.md)。

## Caveat: N=80 統計侷限

`test_ratio=0.2` 後每 class 只剩 2 張 test 影像，總 N=80。單一預測 = 1.25 個百分點。表中 5–10pp 的差距還在 8–9 筆預測以內，對「跨 dataset 比較」雖然方向性正確，但**不適合宣告嚴格單調趨勢**。可靠的結論：「在所有去識別化參數下 CNN Top-1 都遠優於 1/40 隨機猜測」。

## 訓練曲線

每組 100 epoch 的 train/val/test loss 與 accuracy 曲線見：

```
plots/original_{loss,accuracy}.png
plots/pix_b{2,4,8,16}_{loss,accuracy}.png
plots/blur_k{15,45,99}_{loss,accuracy}.png
```

共 16 張 PNG，由 `scripts/plot_log.py --log-dir logs --output-dir plots` 產出。
