# Step 2 CNN Attack Results

本頁整理 CNN 對原圖與去識別化影像的重識別攻擊結果。所有資料集共用 `outputs/split_train.json` 與 `outputs/split_test.json`，每一種去識別化參數都獨立訓練與評估一個 CNN，沒有混合訓練。

評估指令：

```bash
uv run --extra attack python scripts/evaluate.py --all --device auto --output reports/evaluation.csv
```

## Top-1 / Top-5 Accuracy

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

## Analysis

隨機猜測 40 個人的 Top-1 accuracy 約為 1/40 = 0.025，Top-5 accuracy 約為 5/40 = 0.125。實驗結果顯示，即使經過 Pixelization 或 Gaussian Blur，CNN 仍能取得遠高於隨機猜測的重識別準確率，代表一般去識別化處理無法完全阻止 AI 攻擊。

Gaussian Blur 的 k 值變大時，Top-1 accuracy 從 `blur_k15` 的 0.9125 降到 `blur_k99` 的 0.8625，符合模糊強度提高後臉部細節變少、辨識難度增加的直覺。

Pixelization 在這次 AT&T (ORL) 小型資料集上沒有呈現嚴格單調下降。可能原因是 ORL 影像背景、姿態與光照較固定，強 pixelization 仍保留臉型、亮暗分布與髮型等身份線索；此外 test set 只有 80 張，少量樣本容易讓不同參數之間有小幅波動。整體而言，Pixelization 並沒有讓攻擊失效。

目前最低 Top-1 accuracy 是 `blur_k99` 的 0.8625，但仍明顯高於隨機猜測，因此 Step 3 的 DP 方法需要進一步降低可重識別性。
