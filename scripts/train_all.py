"""Train one independent CNN for each original / pixelized / blurred dataset."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DATASETS = {
    "original": "data/att_faces",
    "pix_b2": "data/deid/pixelized/pix_b2",
    "pix_b4": "data/deid/pixelized/pix_b4",
    "pix_b8": "data/deid/pixelized/pix_b8",
    "pix_b16": "data/deid/pixelized/pix_b16",
    "blur_k15": "data/deid/blurred/blur_k15",
    "blur_k45": "data/deid/blurred/blur_k45",
    "blur_k99": "data/deid/blurred/blur_k99",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all Step 2 attack models independently.")
    parser.add_argument("--config", default="config.yaml", help="Training config path.")
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS), help="Dataset names to train.")
    parser.add_argument(
        "--device",
        default=None,
        choices=["auto", "cuda", "mps", "cpu"],
        help="Training device passed to scripts/train.py.",
    )
    args = parser.parse_args()

    for dataset_name in args.datasets:
        if dataset_name not in DATASETS:
            raise ValueError(f"Unknown dataset {dataset_name!r}. Choices: {', '.join(DATASETS)}")

        dataset_root = Path(DATASETS[dataset_name])
        if not dataset_root.exists():
            print(f"skip missing dataset: {dataset_root}")
            continue

        command = [
            sys.executable,
            "scripts/train.py",
            "--dataset-root",
            str(dataset_root),
            "--name",
            dataset_name,
            "--config",
            args.config,
        ]
        if args.device:
            command.extend(["--device", args.device])

        print(f"training {dataset_name}")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
