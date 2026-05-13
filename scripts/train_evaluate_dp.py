"""Train and evaluate CNN attacks on DP datasets from for_cnn.zip.

Expected input layout:

    outputs/for_cnn/
      dp_pix_b8/eps0.1/s1/1.png
      dp_pix_b16/eps0.1/s1/1.png
      dp_blur/eps0.1/s1/1.png

Each DP variant is trained independently, matching the Step 2 attack protocol.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from evaluate import evaluate_one, print_table, write_csv
from train import get_device, load_config


METHODS = {
    "dp_pix_b8": "dp_pix_b8",
    "dp_pix_b16": "dp_pix_b16",
    "dp_blur": "dp_blur",
}
EPSILONS = ["0.1", "0.3", "0.5", "0.7", "1.0", "3.0", "5.0"]


def safe_epsilon(epsilon: str) -> str:
    return epsilon.replace(".", "_")


def build_dp_datasets(root: str | Path) -> dict[str, Path]:
    root = Path(root)
    datasets: dict[str, Path] = {}
    for method_name, method_dir in METHODS.items():
        for epsilon in EPSILONS:
            name = f"{method_name}_eps{safe_epsilon(epsilon)}"
            datasets[name] = root / method_dir / f"eps{epsilon}"
    return datasets


def train_one(name: str, dataset_root: Path, config_path: str, device: str | None) -> None:
    command = [
        sys.executable,
        "scripts/train.py",
        "--dataset-root",
        str(dataset_root),
        "--name",
        name,
        "--config",
        config_path,
    ]
    if device:
        command.extend(["--device", device])
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate all DP CNN attack datasets.")
    parser.add_argument("--root", default="outputs/for_cnn", help="Extracted for_cnn root.")
    parser.add_argument("--config", default="config.yaml", help="Training config path.")
    parser.add_argument("--output", default="reports/dp_evaluation.csv", help="Evaluation CSV output.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional DP dataset names to run.")
    parser.add_argument("--skip-train", action="store_true", help="Only evaluate existing checkpoints.")
    parser.add_argument("--skip-eval", action="store_true", help="Only train checkpoints.")
    parser.add_argument(
        "--device",
        default=None,
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device passed to train.py and evaluate.py.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(args.device or str(config.get("device", "auto")))
    train_split = config["train_split"]
    test_split = config["test_split"]
    checkpoint_dir = Path(str(config.get("checkpoint_dir", "checkpoints")))

    all_datasets = build_dp_datasets(args.root)
    selected_names = args.datasets or list(all_datasets)

    rows: list[dict[str, Any]] = []
    for name in selected_names:
        if name not in all_datasets:
            raise ValueError(f"Unknown dataset {name!r}. Choices: {', '.join(all_datasets)}")

        dataset_root = all_datasets[name]
        if not dataset_root.exists():
            raise FileNotFoundError(f"Missing DP dataset root: {dataset_root}")

        checkpoint_path = checkpoint_dir / f"{name}.pth"
        if not args.skip_train:
            print(f"training {name}")
            train_one(name, dataset_root, args.config, args.device)

        if not args.skip_eval:
            print(f"evaluating {name}")
            rows.append(
                evaluate_one(
                    name=name,
                    dataset_root=dataset_root,
                    checkpoint_path=checkpoint_path,
                    config=config,
                    device=device,
                    train_split=train_split,
                    test_split=test_split,
                )
            )

    if rows:
        write_csv(args.output, rows)
        print_table(rows)
        print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
