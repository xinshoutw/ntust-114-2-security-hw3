"""facedeid: face de-identification helpers for HW3."""

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
