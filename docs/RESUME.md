# 接手指令：完成 DP CNN 攻擊訓練

訓練暫停於另一台機器，本機 Mac mini 接手繼續。

## 目前狀態

- ✅ Stage 1 — Step 2 baseline CNN 全部重訓完成（8 個 checkpoint），使用 `validation set` 挑 best epoch、`test set` 純 held-out。
- ⏸️ Stage 2/3 — DP 影像資料集已產出（共 42 組），CNN 攻擊只訓練了 11 個（DP-Pix-b8/b16/LP-Blur/DP-Blur-Split 各 ε=0.1, 0.3, 0.5 的部分）。剩 31 個待訓。

## 你需要做的事（單一命令）

```bash
# 1) 確認環境
git pull
uv sync --extra attack

# 2) 完成所有剩餘 DP CNN 訓練 + 全部 42 組 test-set 評估
uv run --extra attack python scripts/train_evaluate_dp.py \
    --device auto --skip-existing \
    --output reports/dp_evaluation.csv

# 3) 重畫 figures/dp_metrics_curves.png 與 dp_attack_accuracy.png
uv run --extra attack python scripts/plot_curves.py \
    --metrics data/dp/metrics.csv \
    --attack-csv reports/dp_evaluation.csv

# 4) 確認沒漏東西
uv run pytest                                       # 8 個 smoke test
uv run --extra attack python scripts/evaluate.py --all --device auto  # 重新評估 Step 2 baseline

# 5) 提交、推回
git status
git add -A
git commit -m "chore(checkpoints): finish DP CNN attack training (28 attacks)"
git push
```

預期時間：在 M-series MPS 上約 **25–30 分鐘**（31 個模型 × ~1 分鐘/模型 = 31 分鐘 + 評估約 1 分鐘 + 重畫圖約 10 秒）。

## 為什麼是 31 個

| 方法族 | 訓練組數 | 當前狀態 |
|---|---:|---|
| `dp_pix_b2` × 7 ε | 7 | 全新訓練 |
| `dp_pix_b4` × 7 ε | 7 | 全新訓練 |
| `dp_pix_b8` × 7 ε | 4 missing（0.1/0.3/0.5 已訓） | 補 4 |
| `dp_pix_b16` × 7 ε | 4 missing（0.1/0.3/0.5 已訓） | 補 4 |
| `lp_blur_k45` × 7 ε | 4 missing（0.1/0.3/0.5 已訓） | 補 4 |
| `dp_blur_split_k45` × 7 ε | 5 missing（0.1/0.3 已訓） | 補 5 |
| **小計** | **28 個應有** | **訓 31 次（含 7+7 b2/b4，加 4+4+4+5 補洞）** |

`--skip-existing` 會跳過已存在的 11 個 checkpoint，所以實際只跑 31 次 training，然後 evaluate 全部 28 個（含已存在的 11 個重新 eval）。

> 注意：總 ε 變體 = 4 個 b 值 × 7 ε（DP-Pix） + 2 個機制 × 7 ε（DP-Blur） = 28 + 14 = **42 個 attack model**，不是 28。我前面寫錯，正確 = **42**。

訂正版表：

| 方法族 | 7 ε | 已訓 |
|---|---:|---:|
| `dp_pix_b2` | 7 | 0 |
| `dp_pix_b4` | 7 | 0 |
| `dp_pix_b8` | 7 | 3 |
| `dp_pix_b16` | 7 | 3 |
| `lp_blur_k45` | 7 | 3 |
| `dp_blur_split_k45` | 7 | 2 |
| **總計** | **42** | **11** |

→ 還要訓練 **31 個** model，評估全部 42 個。

## 訓練中產生的東西

- `checkpoints/<name>.pth`：每個 model 一個
- `logs/<name>.csv`：100 epoch 的 loss/acc trace（含 val 欄）
- `reports/dp_evaluation.csv`：42 row 評估結果（held-out test set）
- 若有跑 plot_curves：`figures/dp_metrics_curves.png`、`figures/dp_attack_accuracy.png`

## 如果中途中斷怎麼辦

`--skip-existing` 是 idempotent 的 — 再跑一次同一個命令會自動接續未完成的部分。安全可重入。

## 完成後我這邊要做的事

1. 寫最終報告（草稿大綱見 `docs/division-of-labor.md`）
2. 整理 README 加 caveats 段（N=80 統計噪音那段）
3. 打包 `TeamName_HW3.zip`
