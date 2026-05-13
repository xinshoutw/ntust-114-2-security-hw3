# data/ — 資料集

## AT&T (ORL) 人臉資料庫（已內附，必要）

**規格：** 40 個 subject × 10 張 = 400 張、92×112 灰階 PGM、8-bit。

`data/att_faces/` 已經放好**真實的 ORL 資料**（`s1` … `s40`，每個資料夾 `1.pgm` … `10.pgm`，外加官方 `ORL_README.txt`），可直接使用。

- **官方說明頁：** <https://cam-orl.co.uk/facedatabase.html>
- **本 repo 取得來源：** <https://www.cl.cam.ac.uk/research/dtg/attarchive/pub/data/att_faces.zip>（Cambridge DTG archive 鏡像）
- 其他鏡像：Kaggle `kasikrit/att-database-of-faces`、`sklearn.datasets.fetch_olivetti_faces()`（後者為 64×64 縮小版，與本 repo 的 92×112 原解析度不同）

```
data/att_faces/
├── ORL_README.txt
├── s1/   1.pgm ... 10.pgm
├── s2/   1.pgm ... 10.pgm
...
└── s40/  1.pgm ... 10.pgm
```

驗證：

```bash
uv run python -c "from facedeid import DatasetIndex; \
  idx = DatasetIndex.from_att('data/att_faces'); print(len(idx), '張、', idx.num_classes, '類')"
# 預期：400 張、 40 類
```

> **重新下載：** `data/att_faces/` 不慎遺失時，從上面的 zip 解壓回去即可；或在沒有網路時用
> `uv run python scripts/make_synthetic_orl.py` 生成結構相容的 synthetic ORL 應急（**正式實驗務必用真實 ORL**，
> 合成資料 subject 間區別性過低，CNN 攻擊準確率會偏離論文結果）。

## FaceScrub（選配，較大規模實驗）

FaceScrub 原始發布僅提供 URL list，建議用 cropped 版本（MegaFace 官方 cropped，或第三方鏡像自行 crawl）。
放成 `data/facescrub/<actor_name>/*.jpg`，再用 `DatasetIndex.from_folders('data/facescrub')` 載入。
含背景影像跑去識別化時要加 `--detect-faces`（先偵測人臉再只處理 bbox 區域）。

## CelebA-by-identity 子集（備案）

依 `identity_CelebA.txt` 把同一 identity 的圖檔分到同一資料夾（`data/celeba/<id>/*.jpg`），
同樣用 `DatasetIndex.from_folders` 載入即可。

---

去識別化後的資料集已直接放在 repo：
- `data/deid/pixelized/pix_b{2,4,8,16}/`、`data/deid/blurred/blur_k{15,45,99}/`：Step 1 產出，可用 `./scripts/run_pixelize.sh` / `./scripts/run_blur.sh` 重建。
- `data/dp/{dp_pix_b8,dp_pix_b16,dp_blur}/eps*/`、`data/dp/metrics.csv`：Step 3 產出，由 `scripts/dp_pixelization.py` / `scripts/dp_blur.py` 產生。
- `data/splits/split_train.json` / `split_test.json`：跨所有變體共用的 stratified 切分（seed=42）。
