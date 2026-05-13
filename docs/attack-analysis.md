# Step 2 CNN 重識別攻擊分析

本文件針對 Step 2 八組 CNN 重識別攻擊結果（見 [`attack-results.md`](attack-results.md)）做進一步分析，是分工表中「攻擊分析文件」的對應交付。`attack-results.md` 只放結果對照表與簡短說明，本文件處理「為什麼」。

實驗設定：AT&T (ORL) 40 人 × 10 張 92×112 灰階人臉，stratified 切成 320 訓練 / 80 測試（`seed=42`）。每一組去識別化參數獨立訓練一個 `SimpleCNN`，100 epoch、SGD（lr=0.01, momentum=0.9）、batch=128、輸入 resize 至 128×128。

---

## 1. 與隨機 baseline 的對比

40 類分類任務的隨機猜測下界：

- Top-1：1/40 = **2.5%**
- Top-5：5/40 = **12.5%**

實驗 Top-1 範圍 **85.00% 到 91.25%**，全部高於隨機 34 倍以上。即使是攻擊得最差的 `blur_k99`（k=99 的高斯模糊，肉眼幾乎看不出五官），Top-1 仍有 85.00%，Top-5 高達 98.75%。

> **這是本作業最核心的結論**：「人眼難辨」不代表「AI 難辨」，傳統 pixelization / Gaussian blur 對 CNN 攻擊**基本上無效**。

## 2. Pixelization：為什麼攻擊準確率不單調下降？

| 變體 | b（區塊邊長） | Top-1 | Top-5 | Test Loss |
|---|---:|---:|---:|---:|
| original | — | 91.25% | 98.75% | 0.3234 |
| pix_b2 | 2 | 88.75% | **100.00%** | 0.4328 |
| pix_b4 | 4 | 91.25% | 98.75% | 0.3491 |
| pix_b8 | 8 | 91.25% | **100.00%** | 0.3469 |
| pix_b16 | 16 | 90.00% | 97.50% | **0.5703** |

直覺上 b 越大、保留的細節越少，攻擊準確率應該越低。但實驗結果**全部 4 組 Top-1 都落在 88.75-91.25% 的 2.5pp 區間內**，沒有任何下降趨勢，甚至最強的 `pix_b16` (90.00%) 還略高於最弱的 `pix_b2` (88.75%)。可能原因如下：

### 2.1 ORL 的結構特性
ORL 是 1990 年代採集的小型實驗室資料集，每個人有 10 張：

- **背景單一**（純黑/深灰）→ 像素化後背景仍然是大塊均勻區域，不會干擾分類
- **姿態與光照受控** → 同一人 10 張差異有限，身分識別主要靠**整體輪廓、髮型、頭顱輪廓**
- **影像本身很小**（92×112）→ b=16 區塊化後仍有 6×7 個區塊，相對於 92×112 還是「有結構」

換句話說，ORL 的個體識別線索集中在**低頻訊號**（人臉大致形狀、髮量分布、膚色深淺），而 pixelization 本質上就是一種低通濾波，**保留了大部分低頻訊號**。CNN 學的就是這些低頻特徵。

### 2.2 為什麼 `pix_b16` 反而沒低於 `pix_b2`？
這很可能是**雜訊範圍內的小波動**，原因：

1. Test set 只有 80 張，1 筆預測對應 **1.25 個百分點**。`pix_b2`、`pix_b16` 之間 1.25pp 的差距只是 1 筆對錯
2. 訓練只跑了一個 seed（`seed=42`），MPS 在 PyTorch 上不完全 deterministic，不同硬體/不同 seed 之間 Top-1 容易抖動 ±2-3pp（本實驗確實觀察到組員 3 與組員 4 兩次 run 之間 pix_b2 從 92.50% 變成 88.75%、blur_k45 從 88.75% 變成 91.25% 的反向漂移）
3. **真正的訊號是「整體沒下降」，不是任何一組的絕對數字**

### 2.3 Test Loss 提供補充訊號
即使 Top-1 看起來都差不多，`pix_b16` 的 test loss (0.57) 明顯高於 `pix_b4`/`pix_b8` (0.35)──這代表 CNN 在強 pixelization 下**雖然仍能猜對最像的那個人**，但**信心明顯下降**。換句話說，CNN 是在「勉強識別」而非「自信識別」。

### 2.4 為什麼 Top-5 動輒接近 100%？
Top-5 比 Top-1 寬鬆很多——只要正確答案在前 5 名就算對。在 40 類中佔前 5 已經把任務變得遠比 Top-1 容易，所以即使影像被破壞，CNN 輸出 logits 分布變平緩，仍很容易把正確類別擠進前 5。`pix_b2` 與 `pix_b8` 都打到 100% 而非 `pix_b16` 反而 97.5%，再次顯示這個指標對小幅波動很不敏感。

### 2.5 結論
Pixelization 對 ORL 上的 CNN 攻擊**不是有效防禦**。觀察到的小幅波動是雜訊，不是趨勢。論文 (Fan, 2018) 也指出，這正是引入差分隱私 (DP-Pixelization) 的動機——傳統 pixelization 沒有理論保證。

## 3. Gaussian Blurring：為什麼最強模糊的攻擊準確率最低？

| 變體 | k（核大小） | Top-1 | Top-5 | Test Loss |
|---|---:|---:|---:|---:|
| original | — | 91.25% | 98.75% | 0.3234 |
| blur_k15 | 15 | 90.00% | 98.75% | 0.4382 |
| blur_k45 | 45 | 91.25% | 100.00% | 0.4785 |
| blur_k99 | 99 | **85.00%** | 98.75% | **0.5482** |

Gaussian blur 的下降幅度比 pixelization 明顯：

- **`blur_k99` 是 8 組中 Top-1 最低**，與 `original` 差 6.25pp
- **Test loss 從原圖 0.32 增至 0.55**（×1.7），模型信心顯著下降
- 但 Top-5 仍維持 98.75%，模型仍能把正確類別擠進前 5

不過 `blur_k15`、`blur_k45` 之間並非嚴格單調（k45 反而略高於 k15）。同樣是 N=80 雜訊──但**最強模糊 (k99) 為 8 組最低值**這個訊號穩定存在於兩次獨立 run。

### 3.1 為什麼比 pixelization 更有效？
Pixelization 的區塊在邊界是斷裂的（每個 b×b 內取平均），但**區塊之間的對比**仍保留──CNN 第一層 conv 容易拾起這種「階梯狀」邊界。Gaussian blur 則是**連續的低通濾波**，沒有人造邊界，破壞更乾淨。當 k=99（核覆蓋幾乎整張臉），高頻細節基本被消除。

### 3.2 為什麼 Top-5 還是這麼高？
即使模糊到 k=99，影像中**頭部整體輪廓、髮量、頭顱在畫面中的位置**仍然清晰可見。這些是「全域」訊息，要徹底破壞需要的不只是模糊，而是引入足以掩蓋輪廓的隨機性──這正是 DP-Blur 在 Step 3 嘗試做的事。

## 3.5 Per-class 觀察：哪些人最難被 CNN 認出？

`scripts/attack_analysis.py` 對每組 checkpoint 計算 40 類 per-class Top-1，並把結果橫向比較。完整資料在 `reports/per_class_summary.csv`，跨資料集熱圖在 `figures/per_class_top1_heatmap.png`，每組的 40×40 confusion matrix 在 `figures/confusion/`。

**整體分佈**（每組 40 類中，per-class Top-1 為 1.0 / 部分對 / 0.0 的數量）：

| Dataset | perfect (1.0) | partial (0<acc<1) | zero (0.0) |
|---|---:|---:|---:|
| original | 36 | 1 | 3 |
| pix_b2 | 32 | 7 | 1 |
| pix_b4 | 34 | 5 | 1 |
| pix_b8 | 33 | 7 | 0 |
| pix_b16 | 33 | 6 | 1 |
| blur_k15 | 33 | 6 | 1 |
| blur_k45 | 34 | 5 | 1 |
| blur_k99 | 30 | 8 | 2 |

每類只有 2 筆測試，所以 per-class 只能是 0.0、0.5 或 1.0。可以看到去識別化強度越高，「完美命中」的類別數量略降（original 36 → blur_k99 30），但下降幅度有限。

**最難的 10 個人**（按 8 組的平均 Top-1 排序，由 `reports/hardest_classes.csv` 輸出）：

| Rank | Class ID | Subject | Mean Top-1 |
|---:|---:|:--|---:|
| 1 | 12 | **s13** | 0.1875 |
| 2 | 34 | s35 | 0.5625 |
| 3 | 4 | s5 | 0.5625 |
| 4 | 39 | s40 | 0.6250 |
| 5 | 32 | s33 | 0.6250 |
| 6 | 27 | s28 | 0.6875 |
| 7 | 0 | s1 | 0.6875 |
| 8 | 15 | s16 | 0.7500 |
| 9 | 9 | s10 | 0.8125 |
| 10 | 35 | s36 | 0.8750 |

**`s13` 是顯著的離群值**——8 組裡平均只有 18.75% Top-1（其他「難」的人也都在 50% 以上）。這表示有些受試者本身就被 CNN 系統性地錯認，與去識別化強度無關。可能原因：

- 該 subject 的 10 張在外觀（眼鏡、表情、頭巾、髮型變化）差異特別大，導致 train/test 之間泛化困難
- 該 subject 在 ORL 中與另一個 subject 形狀相近（看 `figures/confusion/original_confusion.png` 可看 s13 被錯分到哪個 class）
- 訓練的 2-shot per class 環境下，這個 class 的 8 張訓練樣本沒抓到關鍵特徵

**對最終報告的價值**：能說「整體 Top-1 ~90% 不是均勻分布，多數人是 100%，少數幾人（特別是 s13）幾乎全錯」——這是 aggregate metric 看不到的細節，可以強化「平均數會掩蓋個體差異」的論點。

## 4. 統計侷限：N=80 能下多強的結論？

- **解析度**：1 筆預測 = 1.25 個百分點。任何 1-2 個百分點的差距都在雜訊範圍內
- **per-class accuracy 不可靠**：每類只有 2 筆測試（80 / 40），單一錯誤就是 0% 或 50%
- **單一 seed**：本實驗只有 `seed=42`。理想上應跑 5 個 seed 計平均±標準差
- **best-epoch 選擇**：checkpoint 取 test_acc 最高的 epoch（見 `scripts/train.py:341-343`），對 test set 略有「資訊洩漏」。實務上更嚴謹的做法是切出獨立 validation set
- **40 類是小規模識別問題**：相比 LFW 5000 人或 CelebA 10000+ 人，這個任務本身偏簡單，攻擊準確率天花板較高

**可以下的結論**：傳統去識別化方法**整體無法把攻擊準確率打回隨機 baseline**（這個結論在 6 個百分點等級的雜訊下仍然成立）。

**不應該過度解讀**：個別變體之間 1-3 個百分點的差距、pix 不單調的細節原因，需要更大規模或多 seed 才能說死。例如組員 3 與組員 4 兩次獨立 run 之間，`pix_b2` 從 92.50% 抖到 88.75%、`blur_k45` 從 88.75% 抖到 91.25%，這就是典型的 N=80 抖動。

## 5. 為什麼還需要 Step 3 DP？

從上面兩節可以總結：

| 防禦強度 | 方法 | 最低 Top-1 | 距離 random（2.5%） |
|---|---|---:|---:|
| 最弱 | pix_b2 | 88.75% | × 35.5 |
| pixelization 最強 | pix_b16 | 90.00% | × 36 |
| blur 最強 | blur_k99 | **85.00%** | × 34 |

傳統方法的**地板**仍在 85% 以上，離隨機猜測差兩個數量級。Step 3 的 ε-差分隱私引入有**理論保證**的隨機性：以 DP-Pixelization 為例，可在 ε=0.1 時把 `pix_b8` 的攻擊準確率從 91.25% 壓到約 63.75%（見 [`dp-attack-results.md`](dp-attack-results.md)），這是傳統方法不可能達到的下降幅度。

DP 的代價是影像效用下降（MSE 增、SSIM 減）。Step 3 的 ε 掃描就是在量化「隱私（攻擊準確率）vs 效用（MSE/SSIM）」這個 trade-off。

## 6. 訓練曲線觀察

各組 best epoch 集中在 **66–97** 之間（original 最早收斂在 66，blur_k45 最晚在 97），均接近 100 epoch 的訓練上限。這代表：

- 模型在 100 epoch 內未明顯過擬合（test_acc 仍在改進）
- 增加 epoch 數可能略提升所有組的天花板，但**不會改變相對排序**（去識別化越強 → Top-1 越低）的結論
- 若想壓 train/test gap，比較有效的會是資料增強（horizontal flip 等）或更大規模 dataset

訓練曲線 PNG（共 16 張，8 組 × loss/accuracy）見 `plots/` 目錄，由 `scripts/plot_log.py --log-dir logs --output-dir plots` 產出。Best epoch + Top-1/Top-5 摘要在 `reports/summary.csv`。

## 7. 對最終報告的建議

當組員 2 整合最終報告時，本作業 Step 2 的核心訊息可濃縮為三句：

1. **40 類 ORL 上，CNN 對原圖達 90% Top-1 / 98.75% Top-5。**
2. **無論是 pixelization（b=2..16）或 Gaussian blur（k=15..99），CNN Top-1 仍維持 86-93%，等於傳統去識別化對 AI 攻擊基本無效。**
3. **這正是引入 DP-Pixelization / DP-Blur（Step 3）的動機──需要有理論保證的隨機化，才能真正打破低頻身分線索。**

可搭配 `figures/pixelize_comparison.png`（人眼看的視覺對比）+ `attack-results.md` 表格（機器看的攻擊準確率）做「人 vs AI」的並列對照。

---

## 附錄：執行重現

```bash
# 1. 訓練 8 組（M4 Pro MPS 約 10-30 分鐘）
uv run --extra attack python scripts/train_all.py --config config.yaml --device auto

# 2. 評估
uv run --extra attack python scripts/evaluate.py --all --device auto --output reports/evaluation.csv

# 3. 訓練曲線
uv run --extra attack python scripts/plot_log.py --log-dir logs --output-dir plots

# 4. Summary
uv run --extra attack python scripts/summarize_logs.py --log-dir logs --output reports/summary.csv
```
