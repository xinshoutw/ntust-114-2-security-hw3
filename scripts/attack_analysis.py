"""Per-class attack analysis (per-class Top-1, cross-dataset heatmap, hardest classes)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from facedeid.model import SimpleCNN
from evaluate import DATASETS, load_checkpoint
from train import build_loaders, get_device, load_config


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    for images, labels in loader:
        non_blocking = device.type == "cuda"
        images = images.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)
        outputs = model(images)
        top1 = outputs.argmax(dim=1)
        preds.append(top1.cpu().numpy())
        trues.append(labels.cpu().numpy())
    return np.concatenate(preds), np.concatenate(trues)


def per_class_accuracy(
    preds: np.ndarray, trues: np.ndarray, num_classes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (acc, correct, total) arrays of shape (num_classes,)."""
    correct = np.zeros(num_classes, dtype=int)
    total = np.zeros(num_classes, dtype=int)
    for true_label in trues:
        total[true_label] += 1
    for pred, true in zip(preds, trues):
        if pred == true:
            correct[true] += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = np.where(total > 0, correct / np.maximum(total, 1), 0.0)
    return acc, correct, total


def confusion_matrix(preds: np.ndarray, trues: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for pred, true in zip(preds, trues):
        cm[true, pred] += 1
    return cm


def save_per_class_csv(
    name: str,
    accs: np.ndarray,
    correct: np.ndarray,
    total: np.ndarray,
    label_to_name: dict[int, str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["dataset", "class_id", "class_name", "test_count", "correct", "top1_acc"])
        for class_id in range(len(accs)):
            writer.writerow([
                name,
                class_id,
                label_to_name.get(class_id, str(class_id)),
                int(total[class_id]),
                int(correct[class_id]),
                round(float(accs[class_id]), 4),
            ])


def save_confusion_heatmap(cm: np.ndarray, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, cmap="viridis", aspect="equal")
    ax.set_xlabel("Predicted class id")
    ax.set_ylabel("True class id")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_per_class_heatmap(
    acc_matrix: np.ndarray,
    dataset_names: list[str],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(acc_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(range(len(dataset_names)))
    ax.set_yticklabels(dataset_names)
    ax.set_xticks(range(acc_matrix.shape[1]))
    ax.set_xticklabels([str(i) for i in range(acc_matrix.shape[1])], rotation=90, fontsize=7)
    ax.set_xlabel("Class id (s1..s40 → 0..39)")
    ax.set_title("Per-class Top-1 accuracy across 8 attack datasets (green=easy, red=hard)")
    fig.colorbar(im, ax=ax, label="Top-1 accuracy")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def report_hardest_classes(
    acc_matrix: np.ndarray,
    dataset_names: list[str],
    label_to_name: dict[int, str],
    output_path: Path,
    top_n: int = 10,
) -> list[tuple[int, str, float]]:
    mean_acc = acc_matrix.mean(axis=0)
    order = np.argsort(mean_acc)
    hardest: list[tuple[int, str, float]] = []
    for class_id in order[:top_n]:
        hardest.append((
            int(class_id),
            label_to_name.get(int(class_id), str(int(class_id))),
            float(mean_acc[class_id]),
        ))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["rank", "class_id", "class_name", "mean_top1_across_8_datasets"])
        for rank, (class_id, class_name, mean) in enumerate(hardest, start=1):
            writer.writerow([rank, class_id, class_name, round(mean, 4)])
    return hardest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS))
    parser.add_argument("--report-dir", default="reports/per_class")
    # 40x40 confusion matrices look identical across datasets with only 2 test
    # samples/class, so we skip them unless --figure-dir is set explicitly.
    parser.add_argument("--figure-dir", default=None)
    parser.add_argument("--summary-csv", default="reports/per_class_summary.csv",
                        )
    parser.add_argument("--heatmap-png", default="figures/per_class_top1_heatmap.png")
    parser.add_argument("--hardest-csv", default="reports/hardest_classes.csv")
    parser.add_argument("--top-n-hardest", type=int, default=10)
    parser.add_argument("--device", default=None, choices=["auto", "cuda", "mps", "cpu"])
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(args.device or str(config.get("device", "auto")))
    train_split = config["train_split"]
    test_split = config["test_split"]
    checkpoint_dir = Path(args.checkpoint_dir)

    all_rows: list[dict] = []
    acc_matrix_rows: list[np.ndarray] = []
    dataset_names: list[str] = []
    label_to_name_global: dict[int, str] | None = None

    for name in args.datasets:
        if name not in DATASETS:
            raise ValueError(f"Unknown dataset {name!r}. Choices: {', '.join(DATASETS)}")
        dataset_root = DATASETS[name]
        checkpoint_path = checkpoint_dir / f"{name}.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")

        checkpoint = load_checkpoint(checkpoint_path, device)
        _, test_loader, label_to_name = build_loaders(
            dataset_root=dataset_root,
            train_split=train_split,
            test_split=test_split,
            config=config,
            device=device,
        )
        if label_to_name_global is None:
            label_to_name_global = label_to_name

        model = SimpleCNN(num_classes=len(label_to_name)).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        preds, trues = collect_predictions(model, test_loader, device)
        num_classes = len(label_to_name)
        accs, correct, total = per_class_accuracy(preds, trues, num_classes)
        cm = confusion_matrix(preds, trues, num_classes)

        save_per_class_csv(
            name, accs, correct, total, label_to_name,
            Path(args.report_dir) / f"{name}.csv",
        )
        overall_top1 = float((preds == trues).mean())
        if args.figure_dir:
            save_confusion_heatmap(
                cm,
                Path(args.figure_dir) / f"{name}_confusion.png",
                title=f"Confusion matrix — {name} (Top-1 = {overall_top1:.4f})",
            )

        n_perfect = int(np.sum((total > 0) & (accs == 1.0)))
        n_zero = int(np.sum((total > 0) & (accs == 0.0)))
        n_partial = int(np.sum((total > 0) & (accs > 0.0) & (accs < 1.0)))
        print(
            f"{name:<10s} | top1={overall_top1:.4f} | "
            f"per-class perfect: {n_perfect}/{num_classes}, "
            f"zero: {n_zero}/{num_classes}, "
            f"partial: {n_partial}/{num_classes}"
        )

        for class_id in range(num_classes):
            all_rows.append({
                "dataset": name,
                "class_id": class_id,
                "class_name": label_to_name.get(class_id, str(class_id)),
                "test_count": int(total[class_id]),
                "correct": int(correct[class_id]),
                "top1_acc": round(float(accs[class_id]), 4),
            })

        acc_matrix_rows.append(accs)
        dataset_names.append(name)

    if not all_rows:
        raise SystemExit("No datasets analyzed.")
    assert label_to_name_global is not None

    summary_path = Path(args.summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"saved: {summary_path}")

    acc_matrix = np.stack(acc_matrix_rows, axis=0)
    heatmap_path = Path(args.heatmap_png)
    save_per_class_heatmap(acc_matrix, dataset_names, heatmap_path)
    print(f"saved: {heatmap_path}")

    hardest = report_hardest_classes(
        acc_matrix, dataset_names, label_to_name_global,
        Path(args.hardest_csv), top_n=args.top_n_hardest,
    )
    print(f"\nHardest {args.top_n_hardest} classes (lowest mean Top-1 across {len(dataset_names)} datasets):")
    print("  rank | class_id | class_name | mean_top1")
    for rank, (class_id, class_name, mean) in enumerate(hardest, start=1):
        print(f"  {rank:>4d} | {class_id:>8d} | {class_name:<10s} | {mean:.4f}")
    print(f"saved: {args.hardest_csv}")


if __name__ == "__main__":
    main()
