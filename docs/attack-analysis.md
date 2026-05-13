# Step 2 CNN 重識別攻擊分析

> 結果對照表見 [`attack-results.md`](attack-results.md)。本文件處理「為什麼」。

實驗設定：ORL 40 人 × 10 張 92×112 灰階人臉，stratified 切成 240 train / 80 val / 80 test（`seed=42`）。每組去識別化參數獨立訓練一個 `SimpleCNN`，SGD lr=0.01 momentum=0.9、batch=128、100 epoch、輸入 resize 至 128×128。best checkpoint 以 val accuracy 挑選，下面所有 Top-1 都是 held-out test set 數字。

## 與隨機 baseline 對比

40 類分類的隨機下界：Top-1 = 1/40 = 2.5%、Top-5 = 12.5%。

實驗 Top-1 範圍 **76.25%–86.25%**，全部高於隨機 30 倍以上。最弱的 `pix_b16` (76.25%) 仍是隨機的 ×30，最弱的 blur `blur_k99` (81.25%) 是隨機的 ×32。

人眼難辨 ≠ AI 難辨：傳統 pixelization / Gaussian blur 擋不住 CNN 攻擊。

## Pixelization

| 變體 | b | Top-1 | Top-5 | Test Loss |
|---|---:|---:|---:|---:|
| original | — | 85.00% | 97.50% | 0.5542 |
| pix_b2 | 2 | 85.00% | 98.75% | 0.4455 |
| pix_b4 | 4 | 85.00% | 98.75% | 0.5528 |
| pix_b8 | 8 | 82.50% | 100.00% | 0.6559 |
| pix_b16 | 16 | **76.25%** | 97.50% | **0.8002** |

`pix_b16` 比 `original` 低 8.75pp、test loss 升 0.25——是 4 組 pix 中**唯一明顯下降**的變體。`pix_b2`、`pix_b4` 與 `original` 持平（兩者塊太小，幾乎沒改變影像），`pix_b8` 介於中間。

ORL 的識別線索集中在低頻訊號（整體輪廓、髮量、頭顱形狀），而 pixelization 是低通濾波，b 不夠大時保留的低頻資訊仍足以分類。要看到明顯下降需要 b ≥ 8。

## Gaussian Blurring

| 變體 | k | Top-1 | Top-5 | Test Loss |
|---|---:|---:|---:|---:|
| original | — | 85.00% | 97.50% | 0.5542 |
| blur_k15 | 15 | 86.25% | 97.50% | 0.6050 |
| blur_k45 | 45 | 81.25% | 97.50% | 0.7022 |
| blur_k99 | 99 | 81.25% | 97.50% | **0.8301** |

Blur 的下降模式比 pix 一致：k=15 與 original 持平（核太小、影響有限），k=45/99 都降到 81.25%。Top-5 維持在 97.5%，模型仍能把正確答案擠進前 5。

Pixelization 在區塊邊界保留銳利對比，CNN 第一層 conv 容易學到「階梯狀」邊界特徵；Gaussian blur 是連續低通濾波、沒有人造邊界，破壞較乾淨。但即使 k=99 把高頻幾乎抹除，頭顱輪廓、髮量分布等全域訊號仍清晰可見——這是傳統模糊無法摧毀的部分。

## Per-class 觀察

cross-dataset 熱圖在 `figures/per_class_top1_heatmap.png`；完整 per-class 數字在 `reports/per_class_summary.csv`。

最難的 10 個 subject（按 8 個 dataset 平均 Top-1 排序，來自 `reports/hardest_classes.csv`）：

| Rank | Subject | Mean Top-1 |
|---:|:--|---:|
| 1 | s13 | 18.75% |
| 2 | s35 | 25.00% |
| 3 | s1 | 50.00% |
| 4 | s5 | 50.00% |
| 5 | s29 | 50.00% |
| 6 | s28 | 50.00% |
| 7 | s33 | 56.25% |
| 8 | s16 | 62.50% |
| 9 | s10 | 62.50% |
| 10 | s32 | 68.75% |

s13、s35 跨所有 8 個 dataset 都低於 30%。這代表 ORL 中有少數 subject 在這個 6-shot 訓練條件下系統性地被 CNN 錯認，與去識別化強度無關。可能原因：subject 自己的 10 張在外觀（眼鏡、髮型、表情）變化太大；或與另一個 subject 形狀近似（具體錯分對象可從 `reports/per_class/original.csv` 反推）。

40 個人的 Top-1 不是均勻分布——多數人 100%、少數幾人 < 40%。aggregate Top-1 平均值會掩蓋這個個體差異。

## 統計侷限

- N=80 → 1 筆預測 = 1.25pp。任何 1-2pp 的差距都在雜訊內。
- per-class accuracy 每類僅 2 筆 → 只能是 0%、50%、100%。
- 單一 seed=42。理想上應跑多 seed 取平均±std。

可下的結論：**所有變體 Top-1 都遠優於隨機，傳統去識別化不足以擋下 CNN 攻擊。**
不應過度解讀：個別變體之間 < 3pp 的差距、pix 不單調的細節原因。

## 為什麼還需要 Step 3 DP

| 強度 | 方法 | 最低 Top-1 |
|---|---|---:|
| pix 最強 | pix_b16 | 76.25% |
| blur 最強 | blur_k45 / blur_k99 | 81.25% |

傳統方法地板在 76% 以上，離隨機 2.5% 差兩個數量級。

ε-差分隱私引入有理論保證的隨機性。`DP-Pix-b2` 在 ε=0.1 把攻擊壓到 **8.75%**（≈ 隨機），這是傳統 pix/blur 不可能達到的（見 [`dp-attack-results.md`](dp-attack-results.md)）。代價是 utility 大幅下降（SSIM 從 0.43 → 0.012）。Step 3 的 ε scan 就是量化這個 privacy-utility trade-off。

## 訓練曲線

各組 best epoch 集中在 73–100（最晚收斂的 `pix_b4` 與 `blur_k99` 都到 100），均接近 epoch 上限。曲線見 `plots/`、summary 在 `reports/summary.csv`。
