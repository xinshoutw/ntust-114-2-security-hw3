"""最小煙霧測試:確認去識別化函式的形狀/型別/邊界行為正確。"""
from __future__ import annotations

import numpy as np
import pytest

from facedeid import gaussian_blur, pixelize


@pytest.fixture
def gray_img() -> np.ndarray:
    rng = np.random.RandomState(0)
    return rng.randint(0, 256, size=(112, 92), dtype=np.uint8)


def test_pixelize_preserves_shape_and_dtype(gray_img: np.ndarray) -> None:
    out = pixelize(gray_img, b=8)
    assert out.shape == gray_img.shape
    assert out.dtype == gray_img.dtype


def test_pixelize_b1_is_identity(gray_img: np.ndarray) -> None:
    np.testing.assert_array_equal(pixelize(gray_img, b=1), gray_img)


def test_pixelize_larger_b_loses_more_detail(gray_img: np.ndarray) -> None:
    # b 越大 → 與原圖差異越大(去識別化越強)
    diff_small = float(np.mean(np.abs(gray_img.astype(int) - pixelize(gray_img, b=2))))
    diff_large = float(np.mean(np.abs(gray_img.astype(int) - pixelize(gray_img, b=16))))
    assert diff_large > diff_small


def test_pixelize_rejects_b_below_one(gray_img: np.ndarray) -> None:
    with pytest.raises(ValueError):
        pixelize(gray_img, b=0)


def test_gaussian_blur_preserves_shape_and_dtype(gray_img: np.ndarray) -> None:
    out = gaussian_blur(gray_img, k=15)
    assert out.shape == gray_img.shape
    assert out.dtype == gray_img.dtype


def test_gaussian_blur_even_k_is_normalized(gray_img: np.ndarray) -> None:
    # 偶數 k 會被 +1 成奇數;結果應與 k+1 相同
    np.testing.assert_array_equal(gaussian_blur(gray_img, k=44), gaussian_blur(gray_img, k=45))


def test_gaussian_blur_larger_k_loses_more_detail(gray_img: np.ndarray) -> None:
    diff_small = float(np.mean(np.abs(gray_img.astype(int) - gaussian_blur(gray_img, k=15))))
    diff_large = float(np.mean(np.abs(gray_img.astype(int) - gaussian_blur(gray_img, k=99))))
    assert diff_large > diff_small


def test_gaussian_blur_rejects_k_below_one(gray_img: np.ndarray) -> None:
    with pytest.raises(ValueError):
        gaussian_blur(gray_img, k=0)
