"""Evaluate Step 2 CNN attack checkpoints with Top-1 and Top-5 accuracy.

Examples:
    uv run --extra attack python scripts/evaluate.py --all --device auto
    uv run --extra attack python scripts/evaluate.py \
        --dataset-root data/deid/pixelized/pix_b8 \
        --checkpoint checkpoints/pix_b8.pth \
        --name pix_b8
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from facedeid.model import SimpleCNN
from train import build_loaders, get_device, load_config


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


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()

    total_loss = 0.0
    total_top1 = 0
    total_top5 = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            non_blocking = device.type == "cuda"
            images = images.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            predictions = outputs.argmax(dim=1)
            total_top1 += (predictions == labels).sum().item()

            k = min(5, outputs.size(1))
            topk_predictions = outputs.topk(k=k, dim=1).indices
            total_top5 += topk_predictions.eq(labels.view(-1, 1)).any(dim=1).sum().item()

    return {
        "samples": total_samples,
        "test_loss": total_loss / total_samples,
        "top1_acc": total_top1 / total_samples,
        "top5_acc": total_top5 / total_samples,
    }


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    required_keys = {"model_state_dict", "label_to_name"}
    missing = required_keys - set(checkpoint)
    if missing:
        raise KeyError(f"{path} is missing checkpoint keys: {sorted(missing)}")
    return checkpoint


def evaluate_one(
    name: str,
    dataset_root: str | Path,
    checkpoint_path: str | Path,
    config: dict[str, Any],
    device: torch.device,
    train_split: str | Path,
    test_split: str | Path,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path, device)

    _, test_loader, split_label_to_name = build_loaders(
        dataset_root=dataset_root,
        train_split=train_split,
        test_split=test_split,
        config=config,
        device=device,
    )

    checkpoint_label_to_name = {int(k): v for k, v in checkpoint["label_to_name"].items()}
    if checkpoint_label_to_name != split_label_to_name:
        raise ValueError(
            f"Label mapping mismatch for {name}. "
            "The checkpoint and split files must use the same class indices."
        )

    model = SimpleCNN(num_classes=len(split_label_to_name)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate_loader(model, test_loader, nn.CrossEntropyLoss(), device)
    return {
        "dataset": name,
        "dataset_root": str(dataset_root),
        "checkpoint": str(checkpoint_path),
        "samples": metrics["samples"],
        "num_classes": len(split_label_to_name),
        "test_loss": round(float(metrics["test_loss"]), 6),
        "top1_acc": round(float(metrics["top1_acc"]), 6),
        "top5_acc": round(float(metrics["top5_acc"]), 6),
        "best_epoch": checkpoint.get("epoch", ""),
        "checkpoint_best_val_acc": round(
            float(checkpoint.get("best_val_acc", checkpoint.get("best_test_acc", 0.0))),
            6,
        ),
        "device": str(device),
    }


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No evaluation rows to write.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict[str, Any]]) -> None:
    print("dataset,top1_acc,top5_acc,test_loss,samples,best_epoch")
    for row in rows:
        print(
            f"{row['dataset']},{row['top1_acc']:.6f},{row['top5_acc']:.6f},"
            f"{row['test_loss']:.6f},{row['samples']},{row['best_epoch']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CNN attack checkpoints.")
    parser.add_argument("--config", default="config.yaml", help="Config path.")
    parser.add_argument("--name", default=None, help="Dataset name for single-checkpoint evaluation.")
    parser.add_argument("--dataset-root", default=None, help="Dataset root for single-checkpoint evaluation.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path for single-checkpoint evaluation.")
    parser.add_argument("--all", action="store_true", help="Evaluate all default Step 2 datasets.")
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS), help="Dataset names used with --all.")
    parser.add_argument("--output", default="reports/evaluation.csv", help="CSV output path.")
    parser.add_argument("--train-split", default=None, help="Defaults to config train_split.")
    parser.add_argument("--test-split", default=None, help="Defaults to config test_split.")
    parser.add_argument(
        "--device",
        default=None,
        choices=["auto", "cuda", "mps", "cpu"],
        help="Evaluation device. Defaults to config device, then auto.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(args.device or str(config.get("device", "auto")))
    train_split = args.train_split or config["train_split"]
    test_split = args.test_split or config["test_split"]
    checkpoint_dir = Path(str(config.get("checkpoint_dir", "checkpoints")))

    rows: list[dict[str, Any]] = []
    if args.all:
        for dataset_name in args.datasets:
            if dataset_name not in DATASETS:
                raise ValueError(f"Unknown dataset {dataset_name!r}. Choices: {', '.join(DATASETS)}")
            dataset_root = DATASETS[dataset_name]
            checkpoint_path = checkpoint_dir / f"{dataset_name}.pth"
            if not Path(dataset_root).exists():
                raise FileNotFoundError(f"Missing dataset root: {dataset_root}")
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
            rows.append(
                evaluate_one(
                    name=dataset_name,
                    dataset_root=dataset_root,
                    checkpoint_path=checkpoint_path,
                    config=config,
                    device=device,
                    train_split=train_split,
                    test_split=test_split,
                )
            )
    else:
        if not (args.name and args.dataset_root and args.checkpoint):
            raise SystemExit("Single evaluation requires --name, --dataset-root, and --checkpoint, or use --all.")
        rows.append(
            evaluate_one(
                name=args.name,
                dataset_root=args.dataset_root,
                checkpoint_path=args.checkpoint,
                config=config,
                device=device,
                train_split=train_split,
                test_split=test_split,
            )
        )

    write_csv(args.output, rows)
    print_table(rows)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
