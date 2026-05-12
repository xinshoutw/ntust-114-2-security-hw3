"""Plot loss and accuracy curves from Step 2 training CSV logs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_log(log_path: Path) -> dict[str, list[float]]:
    rows: dict[str, list[float]] = {}
    with open(log_path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for field in reader.fieldnames or []:
            rows[field] = []
        for row in reader:
            for key, value in row.items():
                rows[key].append(float(value))
    if not rows:
        raise ValueError(f"No data found in {log_path}")
    return rows


def plot_one_log(log_path: Path, output_dir: Path) -> None:
    log = read_log(log_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    title = log_path.stem

    plt.figure(figsize=(8, 5))
    plt.plot(log["epoch"], log["train_loss"], marker="o", label="Train Loss")
    plt.plot(log["epoch"], log["test_loss"], marker="o", label="Test Loss")
    plt.title(f"{title} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    loss_path = output_dir / f"{title}_loss.png"
    plt.savefig(loss_path, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(log["epoch"], log["train_acc"], marker="o", label="Train Top-1 Acc")
    plt.plot(log["epoch"], log["test_acc"], marker="o", label="Test Top-1 Acc")
    plt.plot(log["epoch"], log["test_top5_acc"], marker="o", label="Test Top-5 Acc")
    plt.title(f"{title} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    acc_path = output_dir / f"{title}_accuracy.png"
    plt.savefig(acc_path, dpi=200)
    plt.close()

    print(f"saved: {loss_path}")
    print(f"saved: {acc_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Step 2 training curves.")
    parser.add_argument("--log", default=None, help="Path to one CSV log.")
    parser.add_argument("--log-dir", default=None, help="Plot all CSV logs in this folder.")
    parser.add_argument("--output-dir", default="plots", help="Output plot directory.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.log:
        plot_one_log(Path(args.log), output_dir)
        return

    if args.log_dir:
        log_paths = sorted(Path(args.log_dir).glob("*.csv"))
        if not log_paths:
            raise FileNotFoundError(f"No CSV logs found in {args.log_dir}")
        for log_path in log_paths:
            plot_one_log(log_path, output_dir)
        return

    raise ValueError("Please provide either --log or --log-dir.")


if __name__ == "__main__":
    main()
