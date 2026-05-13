"""facedeid — 人臉去識別化工具組(HW3)。

模組:
  * dataset_loader : 統一資料載入 / stratified train-test split(seed=42)
  * face_detector  : 人臉偵測(Haar Cascade / dlib HOG+SVM)
  * pixelize       : Pixelization 去識別化(b×b 區塊取平均)
  * gaussian_blur  : Gaussian Blurring 去識別化(k×k 高斯核卷積)
"""

from .dataset_loader import (
    DatasetIndex,
    stratified_split,
    stratified_split_3way,
    load_image,
    iter_dataset,
)
from .face_detector import FaceDetector, BBox, detect_faces
from .pixelize import pixelize, pixelize_region, pixelize_faces
from .gaussian_blur import gaussian_blur, gaussian_blur_region, gaussian_blur_faces

__all__ = [
    "DatasetIndex",
    "stratified_split",
    "stratified_split_3way",
    "load_image",
    "iter_dataset",
    "FaceDetector",
    "BBox",
    "detect_faces",
    "pixelize",
    "pixelize_region",
    "pixelize_faces",
    "gaussian_blur",
    "gaussian_blur_region",
    "gaussian_blur_faces",
]
