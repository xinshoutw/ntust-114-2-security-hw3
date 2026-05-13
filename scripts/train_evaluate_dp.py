"""Train and evaluate CNN attacks on the DP datasets in data/dp/."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from evaluate import evaluate_one, print_table, write_csv
from train import get_device, load_config


EPSILONS = ["0.1", "0.3", "0.5", "0.7", "1.0", "3.0", "5.0"]
PIX_BLOCK_SIZES = (2, 4, 8, 16)
BLUR_K_DEFAULT = 45


def safe_epsilon(epsilon: str) -> str:
    return epsilon.replace(".", "_")


def build_dp_datasets(root: str | Path, blur_k: int = BLUR_K_DEFAULT) -> dict[str, Path]:
    root = Path(root)
    datasets: dict[str, Path] = {}
    for epsilon in EPSILONS:
        safe = safe_epsilon(epsilon)
        for b in PIX_BLOCK_SIZES:
            datasets[f"dp_pix_b{b}_eps{safe}"] = root / f"dp_pix_b{b}" / f"eps{epsilon}"
        datasets[f"lp_blur_k{blur_k}_eps{safe}"] = root / "lp_blur" / f"k{blur_k}_eps{epsilon}"
        datasets[f"dp_blur_split_k{blur_k}_eps{safe}"] = (
            root / "dp_blur_split" / f"k{blur_k}_eps{epsilon}"
        )
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/dp")
    parser.add_argument("--blur-k", type=int, default=BLUR_K_DEFAULT)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="reports/dp_evaluation.csv")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--device", default=None, choices=["auto", "cuda", "mps", "cpu"])
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(args.device or str(config.get("device", "auto")))
    train_split = config["train_split"]
    test_split = config["test_split"]
    checkpoint_dir = Path(str(config.get("checkpoint_dir", "checkpoints")))

    all_datasets = build_dp_datasets(args.root, blur_k=args.blur_k)
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
            if args.skip_existing and checkpoint_path.exists():
                print(f"skip-existing: {name}")
            else:
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
