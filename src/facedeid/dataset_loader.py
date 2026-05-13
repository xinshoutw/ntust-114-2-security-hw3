"""DatasetIndex + stratified train/val/test split for ORL-style folders."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np


IMG_EXTS = {".pgm", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class DatasetIndex:
    items: list[tuple[str, int]] = field(default_factory=list)
    label_to_name: dict[int, str] = field(default_factory=dict)
    root: str = ""

    @classmethod
    def from_att(cls, root: str | Path) -> "DatasetIndex":
        """ORL: root/s1..s40, each with 10 PGMs. Labels: s1 → 0, ..., s40 → 39."""
        root = Path(root)
        assert root.is_dir(), f"ORL root not found: {root}"
        # Numeric sort on the sX suffix so s10 comes after s2.
        subj_dirs = sorted(
            [d for d in root.iterdir() if d.is_dir() and d.name.startswith("s")],
            key=lambda d: int(d.name[1:]),
        )
        assert subj_dirs, f"no sX subdirs under {root}"
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
        """Generic: each subfolder is one class (FaceScrub, CelebA-by-identity, etc)."""
        root = Path(root)
        assert root.is_dir(), f"root not found: {root}"
        cls_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
        assert cls_dirs, f"no class subdirs under {root}"
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
        # JSON converts int keys to str and tuples to lists — restore both.
        data["label_to_name"] = {int(k): v for k, v in data["label_to_name"].items()}
        data["items"] = [(p, int(lbl)) for p, lbl in data["items"]]
        return cls(**data)


def stratified_split(
    idx: DatasetIndex, test_ratio: float = 0.2, seed: int = 42
) -> tuple[DatasetIndex, DatasetIndex]:
    """Stratified train/test split. Each class gets >= 1 sample in test."""
    assert 0 < test_ratio < 1
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
    """Stratified 3-way split. Each class gets >= 1 sample in val and test."""
    assert 0 < val_ratio < 1 and 0 < test_ratio < 1
    assert val_ratio + test_ratio < 1
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
            raise ValueError(f"class {label}: not enough samples (n={n}, val={n_val}, test={n_test})")
        test_items.extend((p, label) for p in paths[:n_test])
        val_items.extend((p, label) for p in paths[n_test : n_test + n_val])
        train_items.extend((p, label) for p in paths[n_test + n_val :])

    label_to_name = dict(idx.label_to_name)
    return (
        DatasetIndex(items=train_items, label_to_name=label_to_name, root=idx.root),
        DatasetIndex(items=val_items, label_to_name=label_to_name, root=idx.root),
        DatasetIndex(items=test_items, label_to_name=label_to_name, root=idx.root),
    )


def load_image(path: str, grayscale: bool = True) -> np.ndarray:
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(path, flag)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return img


def iter_dataset(
    idx: DatasetIndex, grayscale: bool = True
) -> Iterator[tuple[np.ndarray, int, str]]:
    for path, label in idx.items:
        yield load_image(path, grayscale=grayscale), label, path


def load_batch(paths: Iterable[str], grayscale: bool = True) -> np.ndarray:
    return np.stack([load_image(p, grayscale=grayscale) for p in paths], axis=0)


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/att_faces"
    idx = DatasetIndex.from_att(root)
    print(f"loaded {len(idx)} images, {idx.num_classes} classes (root={idx.root})")
    train, test = stratified_split(idx, test_ratio=0.2, seed=42)
    print(f"train: {len(train)}  test: {len(test)}")
