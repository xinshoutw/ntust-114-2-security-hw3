"""
dataset_loader.py
-----------------
統一的資料載入介面,給組員 3、4(訓練/評估 CNN)及組員 5(DP)使用。

支援三種資料集:
  * AT&T (ORL):結構為 ROOT/s1, s2, ..., s40,每個資料夾 10 張 .pgm
  * FaceScrub:結構為 ROOT/<actor_name>/*.jpg|png(若使用者下載 cropped 版)
  * 自訂目錄:任意 ROOT/<class_name>/*.{png,jpg,jpeg,bmp,pgm} 結構

提供:
  - DatasetIndex:列出所有 (image_path, label) 配對
  - stratified_split:依 label 切 train/test
  - load_image / load_batch:讀圖工具
  - iter_dataset:逐張產生 (img_array, label, path)

使用範例(訓練端):
    from facedeid.dataset_loader import DatasetIndex, stratified_split, load_image
    idx = DatasetIndex.from_att("/path/to/att_faces")
    train, test = stratified_split(idx, test_ratio=0.2, seed=42)
    for path, label in train.items:
        img = load_image(path)         # numpy ndarray, uint8
        ...

設計原則:
  - 不假設 PyTorch / TensorFlow,只回傳 numpy + path,讓下游自己包成 DataLoader
  - label 一律從 0 開始的 int(原始 ORL 是 1-indexed,這裡轉成 0-indexed)
  - 切分結果可以序列化(toJSON)讓所有組員的實驗用同一個 split
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np


# ---- 副檔名 -----------------------------------------------------------------
IMG_EXTS = {".pgm", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ---- DatasetIndex -----------------------------------------------------------
@dataclass
class DatasetIndex:
    """一個 dataset 的索引(image path + label)。"""

    items: list[tuple[str, int]] = field(default_factory=list)
    label_to_name: dict[int, str] = field(default_factory=dict)
    root: str = ""

    # ---------- 產生器 ------------------------------------------------------
    @classmethod
    def from_att(cls, root: str | Path) -> "DatasetIndex":
        """
        AT&T (ORL):root/s1, s2, ..., s40,每個 10 張 .pgm。
        label 從 0 開始,對應原始 s1 → 0、s2 → 1、...、s40 → 39。
        """
        root = Path(root)
        assert root.is_dir(), f"找不到 ORL 根目錄:{root}"
        # 用 sX 後綴的數字排序,而非字串排序(避免 s10 排到 s2 前面)
        subj_dirs = sorted(
            [d for d in root.iterdir() if d.is_dir() and d.name.startswith("s")],
            key=lambda d: int(d.name[1:]),
        )
        assert subj_dirs, f"{root} 底下沒看到任何 sX 資料夾"
        items: list[tuple[str, int]] = []
        label_to_name: dict[int, str] = {}
        for label, sd in enumerate(subj_dirs):
            label_to_name[label] = sd.name
            for f in sorted(sd.iterdir()):
                if f.suffix.lower() in IMG_EXTS:
                    items.append((str(f), label))
        return cls(items=items, label_to_name=label_to_name, root=str(root))

    @classmethod
    def from_folders(cls, root: str | Path) -> "DatasetIndex":
        """
        通用版本:每個子資料夾 = 一個 class。
        適用 FaceScrub、CelebA-by-identity、自製資料集。
        """
        root = Path(root)
        assert root.is_dir(), f"找不到資料根目錄:{root}"
        cls_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
        assert cls_dirs, f"{root} 底下沒有任何 class 資料夾"
        items: list[tuple[str, int]] = []
        label_to_name: dict[int, str] = {}
        for label, cd in enumerate(cls_dirs):
            label_to_name[label] = cd.name
            for f in sorted(cd.rglob("*")):
                if f.is_file() and f.suffix.lower() in IMG_EXTS:
                    items.append((str(f), label))
        return cls(items=items, label_to_name=label_to_name, root=str(root))

    # ---------- 屬性 --------------------------------------------------------
    @property
    def num_classes(self) -> int:
        return len(self.label_to_name)

    def __len__(self) -> int:
        return len(self.items)

    # ---------- 序列化 ------------------------------------------------------
    def save_json(self, path: str | Path) -> None:
        """把 split 結果存成 json,讓所有組員共用同一份切分。"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> "DatasetIndex":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # JSON 把 int key 轉成 str,要轉回來
        data["label_to_name"] = {int(k): v for k, v in data["label_to_name"].items()}
        # tuple 也被轉成 list,還原
        data["items"] = [(p, int(lbl)) for p, lbl in data["items"]]
        return cls(**data)


# ---- split ------------------------------------------------------------------
def stratified_split(
    idx: DatasetIndex, test_ratio: float = 0.2, seed: int = 42
) -> tuple[DatasetIndex, DatasetIndex]:
    """
    依 label 做 stratified split。每個 class 都至少有 1 張在 test。
    回傳 (train_index, test_index),兩者共用同一個 label_to_name。

    例:ORL 每 class 有 10 張,test_ratio=0.2 → 2 張 test、8 張 train。
    """
    assert 0 < test_ratio < 1, "test_ratio 必須在 (0, 1) 之間"
    rng = random.Random(seed)
    by_label: dict[int, list[str]] = {}
    for path, label in idx.items:
        by_label.setdefault(label, []).append(path)

    train_items, test_items = [], []
    for label, paths in by_label.items():
        paths = list(paths)
        rng.shuffle(paths)
        n_test = max(1, int(round(len(paths) * test_ratio)))
        test_items.extend((p, label) for p in paths[:n_test])
        train_items.extend((p, label) for p in paths[n_test:])

    train = DatasetIndex(items=train_items, label_to_name=dict(idx.label_to_name), root=idx.root)
    test = DatasetIndex(items=test_items, label_to_name=dict(idx.label_to_name), root=idx.root)
    return train, test


def stratified_split_3way(
    idx: DatasetIndex,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[DatasetIndex, DatasetIndex, DatasetIndex]:
    """
    Stratified 3-way split: train / val / test。每個 class 都至少 1 張在 val、1 張在 test。

    回傳 (train, val, test),三者共用同一個 label_to_name。

    例：ORL 每 class 10 張、val_ratio=0.2、test_ratio=0.2 → 每 class 6 train、2 val、2 test。
    """
    assert 0 < val_ratio < 1 and 0 < test_ratio < 1, "val/test ratio 必須在 (0, 1)"
    assert val_ratio + test_ratio < 1, "val + test 不可 >= 1,要留資料給 train"
    rng = random.Random(seed)
    by_label: dict[int, list[str]] = {}
    for path, label in idx.items:
        by_label.setdefault(label, []).append(path)

    train_items, val_items, test_items = [], [], []
    for label, paths in by_label.items():
        paths = list(paths)
        rng.shuffle(paths)
        n = len(paths)
        n_test = max(1, int(round(n * test_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        if n_test + n_val >= n:
            raise ValueError(
                f"class {label} 樣本不夠分:n={n}, n_val={n_val}, n_test={n_test}"
            )
        test_items.extend((p, label) for p in paths[:n_test])
        val_items.extend((p, label) for p in paths[n_test : n_test + n_val])
        train_items.extend((p, label) for p in paths[n_test + n_val :])

    label_to_name = dict(idx.label_to_name)
    return (
        DatasetIndex(items=train_items, label_to_name=label_to_name, root=idx.root),
        DatasetIndex(items=val_items, label_to_name=label_to_name, root=idx.root),
        DatasetIndex(items=test_items, label_to_name=label_to_name, root=idx.root),
    )


# ---- load -------------------------------------------------------------------
def load_image(path: str, grayscale: bool = True) -> np.ndarray:
    """讀單張圖。預設灰階(因為 ORL 是灰階,實驗統一灰階較方便)。"""
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(path, flag)
    if img is None:
        raise FileNotFoundError(f"讀不到圖:{path}")
    return img


def iter_dataset(
    idx: DatasetIndex, grayscale: bool = True
) -> Iterator[tuple[np.ndarray, int, str]]:
    """逐張迭代 (img, label, path)。給批次處理用。"""
    for path, label in idx.items:
        yield load_image(path, grayscale=grayscale), label, path


def load_batch(
    paths: Iterable[str], grayscale: bool = True
) -> np.ndarray:
    """讀一批圖回 numpy array(N, H, W) 或 (N, H, W, 3)。要求所有圖同尺寸。"""
    imgs = [load_image(p, grayscale=grayscale) for p in paths]
    return np.stack(imgs, axis=0)


# ---- CLI smoke test --------------------------------------------------------
if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/att_faces"
    idx = DatasetIndex.from_att(root)
    print(f"載入 {len(idx)} 張、{idx.num_classes} 類 (root={idx.root})")
    train, test = stratified_split(idx, test_ratio=0.2, seed=42)
    print(f"train: {len(train)} 張、test: {len(test)} 張")
    img, lab, p = next(iter_dataset(test))
    print(f"sample test: shape={img.shape} label={lab} ({idx.label_to_name[lab]}) path={p}")
