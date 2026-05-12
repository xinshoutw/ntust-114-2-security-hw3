"""
face_detector.py
----------------
人臉偵測模組,輸入一張影像、輸出人臉的 bounding box 列表。

提供兩種後端,作業 PDF 兩個都列為可用:
  * "haar" (預設):OpenCV Haar Cascade,內建、無外部依賴,正面臉效果夠用
  * "hog"        :dlib HOG + SVM,精度較高、但需要 pip install dlib

兩種偵測器都統一回傳 List[BBox],BBox 的座標是 (x, y, w, h),(x, y) 為左上角。

特殊情況處理:
  - 對於 AT&T (ORL) 這種已經 cropped 的人臉(整張就是臉),偵測器可能因為臉
    占滿整張圖而抓不到;這時就 fallback 到「整張圖視為一個 bbox」(這是組員 1
    所有去識別化方法都該支援的情境,因為對 ORL 我們本來就是對全圖做 pixelization
    或 blur)。

使用範例:
    from facedeid.face_detector import FaceDetector
    det = FaceDetector(backend="haar")
    boxes = det.detect(img)             # List[(x, y, w, h)]
    or
    boxes = det.detect(img, fallback_full=True)   # 偵測不到 → 回 [整張圖]
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np


BBox = tuple[int, int, int, int]   # (x, y, w, h)


# ---- 主要類別 ---------------------------------------------------------------
@dataclass
class FaceDetector:
    backend: Literal["haar", "hog"] = "haar"

    # Haar 參數
    scale_factor: float = 1.1
    min_neighbors: int = 4
    min_size: tuple[int, int] = (20, 20)

    # 內部 lazy init
    _haar: cv2.CascadeClassifier | None = None
    _hog = None  # dlib.fhog_object_detector

    def __post_init__(self) -> None:
        if self.backend == "haar":
            self._haar = self._load_haar()
        elif self.backend == "hog":
            self._hog = self._load_hog()
        else:
            raise ValueError(f"未知 backend: {self.backend!r},允許 'haar' | 'hog'")

    # ---------- 載入後端 ----------------------------------------------------
    @staticmethod
    def _load_haar() -> cv2.CascadeClassifier:
        # OpenCV 內建在 cv2.data.haarcascades
        xml_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not xml_path.exists():
            raise FileNotFoundError(f"找不到 haar cascade:{xml_path}")
        clf = cv2.CascadeClassifier(str(xml_path))
        if clf.empty():
            raise RuntimeError("haar cascade 載入失敗")
        return clf

    @staticmethod
    def _load_hog():
        try:
            import dlib  # 只在用到時 import,避免無 dlib 環境也能用 haar
        except ImportError as e:
            raise ImportError(
                "需要 dlib:pip install dlib(編譯需要 cmake,可改用 backend='haar')"
            ) from e
        return dlib.get_frontal_face_detector()

    # ---------- 偵測 --------------------------------------------------------
    def detect(self, img: np.ndarray, fallback_full: bool = False) -> list[BBox]:
        """
        偵測人臉。img 為 numpy array,灰階 (H, W) 或彩色 (H, W, 3) 都可。

        fallback_full=True 時,若偵測不到任何臉,回傳整張圖當作 bbox
        (對 ORL 這種 cropped 人臉 dataset 很有用)。
        """
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.ndim == 2:
            gray = img
        else:
            raise ValueError(f"img.ndim 必須是 2 或 3,得到 {img.ndim}")

        if self.backend == "haar":
            assert self._haar is not None
            faces = self._haar.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size,
            )
            boxes = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
        else:  # hog
            assert self._hog is not None
            rects = self._hog(gray, 1)   # 1 = upsample 一次,小臉也抓得到
            boxes = []
            for r in rects:
                x, y = max(0, r.left()), max(0, r.top())
                w, h = r.right() - x, r.bottom() - y
                # clip 到圖內
                w = min(w, gray.shape[1] - x)
                h = min(h, gray.shape[0] - y)
                boxes.append((int(x), int(y), int(w), int(h)))

        if not boxes and fallback_full:
            H, W = gray.shape[:2]
            return [(0, 0, W, H)]
        return boxes

    # ---------- 視覺化(debug 用) ------------------------------------------
    @staticmethod
    def draw(img: np.ndarray, boxes: list[BBox], color=(0, 255, 0), thickness: int = 2) -> np.ndarray:
        """在影像上把 bbox 畫出來。回傳 BGR copy。"""
        if img.ndim == 2:
            out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            out = img.copy()
        for (x, y, w, h) in boxes:
            cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
        return out


# ---- 便利函式 ---------------------------------------------------------------
def detect_faces(
    img: np.ndarray, backend: str = "haar", fallback_full: bool = False
) -> list[BBox]:
    """一行就能偵測的便利包裝(每次呼叫會 init detector,小批量適用)。"""
    return FaceDetector(backend=backend).detect(img, fallback_full=fallback_full)


# ---- CLI smoke test --------------------------------------------------------
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/att_faces/s1/1.pgm"
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    det = FaceDetector(backend="haar")
    boxes = det.detect(img, fallback_full=True)
    print(f"{path}: {len(boxes)} face(s) -> {boxes}")
    out = det.draw(img, boxes)
    out_path = "/tmp/face_detect_demo.png"
    cv2.imwrite(out_path, out)
    print(f"視覺化結果寫到:{out_path}")
