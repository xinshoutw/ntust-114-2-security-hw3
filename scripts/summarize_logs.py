"""Summarize best Top-1 / Top-5 accuracy from Step 2 training logs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(log_path: Path) -> list[dict[str, str]]:
    with open(log_path, "r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def summarize_log(log_path: Path, metric: str) -> dict[str, str | float | int]:
    rows = read_rows(log_path)
    if not rows:
        raise ValueError(f"{log_path} has no rows.")
    best = max(rows, key=lambda row: float(row[metric]))
    return {
        "dataset": log_path.stem,
        "best_epoch": int(float(best["epoch"])),
        "best_metric": metric,
        "train_loss": float(best["train_loss"]),
        "train_acc": float(best["train_acc"]),
        "test_loss": float(best["test_loss"]),
        "top1_acc": float(best["test_acc"]),
        "top5_acc": float(best["test_top5_acc"]),
    }


def write_summary(rows: list[dict[str, str | float | int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "best_epoch",
        "best_metric",
        "train_loss",
        "train_acc",
        "test_loss",
        "top1_acc",
        "top5_acc",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Step 2 training logs.")
    parser.add_argument("--log-dir", default="logs", help="Folder containing CSV logs.")
    parser.add_argument("--output", default="reports/summary.csv", help="Output summary CSV.")
    parser.add_argument("--metric", default="test_acc", choices=["test_acc", "test_top5_acc"])
    args = parser.parse_args()

    log_paths = sorted(Path(args.log_dir).glob("*.csv"))
    if not log_paths:
        raise FileNotFoundError(f"No CSV logs found in {args.log_dir}")

    rows = [summarize_log(log_path, args.metric) for log_path in log_paths]
    write_summary(rows, Path(args.output))

    print("dataset,best_epoch,top1_acc,top5_acc")
    for row in rows:
        print(f"{row['dataset']},{row['best_epoch']},{row['top1_acc']:.6f},{row['top5_acc']:.6f}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
