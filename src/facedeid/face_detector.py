"""Face detection: Haar cascade (default) or dlib HOG+SVM."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np


BBox = tuple[int, int, int, int]


@dataclass
class FaceDetector:
    backend: Literal["haar", "hog"] = "haar"
    scale_factor: float = 1.1
    min_neighbors: int = 4
    min_size: tuple[int, int] = (20, 20)
    _haar: cv2.CascadeClassifier | None = None
    _hog = None

    def __post_init__(self) -> None:
        if self.backend == "haar":
            self._haar = self._load_haar()
        elif self.backend == "hog":
            self._hog = self._load_hog()
        else:
            raise ValueError(f"unknown backend: {self.backend!r}")

    @staticmethod
    def _load_haar() -> cv2.CascadeClassifier:
        xml_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not xml_path.exists():
            raise FileNotFoundError(f"haar cascade not found: {xml_path}")
        clf = cv2.CascadeClassifier(str(xml_path))
        if clf.empty():
            raise RuntimeError("haar cascade load failed")
        return clf

    @staticmethod
    def _load_hog():
        try:
            import dlib
        except ImportError as e:
            raise ImportError("dlib required for HOG backend (pip install dlib)") from e
        return dlib.get_frontal_face_detector()

    def detect(self, img: np.ndarray, fallback_full: bool = False) -> list[BBox]:
        """When fallback_full=True and no face is detected, return the whole image as one bbox."""
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.ndim == 2:
            gray = img
        else:
            raise ValueError(f"img.ndim must be 2 or 3, got {img.ndim}")

        if self.backend == "haar":
            assert self._haar is not None
            faces = self._haar.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size,
            )
            boxes = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
        else:
            assert self._hog is not None
            rects = self._hog(gray, 1)
            boxes = []
            for r in rects:
                x, y = max(0, r.left()), max(0, r.top())
                w, h = r.right() - x, r.bottom() - y
                w = min(w, gray.shape[1] - x)
                h = min(h, gray.shape[0] - y)
                boxes.append((int(x), int(y), int(w), int(h)))

        if not boxes and fallback_full:
            H, W = gray.shape[:2]
            return [(0, 0, W, H)]
        return boxes

    @staticmethod
    def draw(img: np.ndarray, boxes: list[BBox], color=(0, 255, 0), thickness: int = 2) -> np.ndarray:
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
        for (x, y, w, h) in boxes:
            cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
        return out


def detect_faces(img: np.ndarray, backend: str = "haar", fallback_full: bool = False) -> list[BBox]:
    return FaceDetector(backend=backend).detect(img, fallback_full=fallback_full)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/att_faces/s1/1.pgm"
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    boxes = FaceDetector(backend="haar").detect(img, fallback_full=True)
    print(f"{path}: {len(boxes)} face(s) -> {boxes}")
