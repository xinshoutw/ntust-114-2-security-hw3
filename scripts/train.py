"""Train one CNN classifier on one dataset variant.

Example:
    uv run --extra attack python scripts/train.py \
        --dataset-root outputs/pixelized/pix_b8 \
        --name pix_b8 \
        --config config.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from facedeid.model import SimpleCNN


IMG_EXTS = [".png", ".pgm", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def load_config(path: str | Path) -> dict[str, Any]:
    """Read the simple key-value config.yaml used by this homework.

    The file only uses scalar values, so a small parser avoids adding another
    dependency outside the existing uv optional attack group.
    """
    config: dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            key, value = line.split(":", 1)
            value = value.strip()
            if value.lower() in {"true", "false"}:
                parsed: Any = value.lower() == "true"
            else:
                try:
                    parsed = int(value)
                except ValueError:
                    try:
                        parsed = float(value)
                    except ValueError:
                        parsed = value.strip("\"'")
            config[key.strip()] = parsed
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(requested: str = "auto") -> torch.device:
    """Return the training device.

    auto:
        Prefer CUDA, then Apple Metal/MPS, then CPU.
    cuda/mps:
        Require that GPU backend. If it is unavailable, fail loudly instead of
        silently falling back to CPU.
    cpu:
        Force CPU training.
    """
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build or use --device auto/cpu."
            )
        return torch.device("cuda")

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS was requested, but torch.backends.mps.is_available() is False. "
                "Run this on an Apple Silicon Mac with an MPS-enabled PyTorch build "
                "or use --device auto/cpu."
            )
        return torch.device("mps")

    if requested == "cpu":
        return torch.device("cpu")

    raise ValueError("device must be one of: auto, cuda, mps, cpu")


def load_split(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def map_split_path(dataset_root: Path, split_path: str) -> Path:
    """Map a split path like data/att_faces/s1/2.pgm to the chosen dataset root.

    Step 1 outputs preserve the ORL subject/file structure but may change the
    extension from .pgm to .png. This function keeps the subject and stem, then
    finds the existing file under dataset_root.
    """
    original = Path(split_path)
    subject = original.parent.name
    stem = original.stem

    candidates = [dataset_root / subject / f"{stem}{original.suffix}"]
    candidates.extend(dataset_root / subject / f"{stem}{ext}" for ext in IMG_EXTS)
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Cannot map split path {split_path!r} into dataset root {dataset_root}"
    )


class SplitImageDataset(Dataset):
    """PyTorch Dataset backed by outputs/split_train.json or split_test.json."""

    def __init__(self, dataset_root: str | Path, split_json: str | Path, image_size: int):
        self.dataset_root = Path(dataset_root)
        split = load_split(split_json)
        self.items = [
            (map_split_path(self.dataset_root, path), int(label))
            for path, label in split["items"]
        ]
        self.label_to_name = {int(k): v for k, v in split["label_to_name"].items()}
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.items[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), label


def build_loaders(
    dataset_root: str | Path,
    train_split: str | Path,
    test_split: str | Path,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[DataLoader, DataLoader, dict[int, str]]:
    train_dataset = SplitImageDataset(dataset_root, train_split, int(config["image_size"]))
    test_dataset = SplitImageDataset(dataset_root, test_split, int(config["image_size"]))

    if train_dataset.label_to_name != test_dataset.label_to_name:
        raise ValueError("Train/test split files must use the same labels.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    return train_loader, test_loader, train_dataset.label_to_name


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
) -> tuple[float, float, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_top5_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(is_training):
        for images, labels in loader:
            non_blocking = device.type == "cuda"
            images = images.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()

            k = min(5, outputs.size(1))
            top5_predictions = outputs.topk(k=k, dim=1).indices
            total_top5_correct += top5_predictions.eq(labels.view(-1, 1)).any(dim=1).sum().item()

    return (
        total_loss / total_samples,
        total_correct / total_samples,
        total_top5_correct / total_samples,
    )


def write_log_header(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "test_top5_acc"])


def append_log(log_path: Path, row: list[float | int]) -> None:
    with open(log_path, "a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(row)


def save_checkpoint(
    output_path: Path,
    model: nn.Module,
    label_to_name: dict[int, str],
    config: dict[str, Any],
    epoch: int,
    best_test_acc: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_to_name": label_to_name,
            "config": config,
            "epoch": epoch,
            "best_test_acc": best_test_acc,
        },
        output_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CNN on one de-identified dataset.")
    parser.add_argument("--dataset-root", required=True, help="Dataset root, e.g. outputs/pixelized/pix_b8.")
    parser.add_argument("--name", required=True, help="Name used for checkpoint/log files.")
    parser.add_argument("--config", default="config.yaml", help="Training config path.")
    parser.add_argument("--train-split", default=None, help="Defaults to config train_split.")
    parser.add_argument("--test-split", default=None, help="Defaults to config test_split.")
    parser.add_argument(
        "--device",
        default=None,
        choices=["auto", "cuda", "mps", "cpu"],
        help="Training device. Defaults to config device, then auto.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    train_split = args.train_split or config["train_split"]
    test_split = args.test_split or config["test_split"]
    checkpoint_dir = Path(str(config.get("checkpoint_dir", "checkpoints")))
    log_dir = Path(str(config.get("log_dir", "logs")))

    device = get_device(args.device or str(config.get("device", "auto")))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    train_loader, test_loader, label_to_name = build_loaders(
        dataset_root=args.dataset_root,
        train_split=train_split,
        test_split=test_split,
        config=config,
        device=device,
    )

    model = SimpleCNN(num_classes=len(label_to_name)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=float(config["learning_rate"]),
        momentum=float(config["momentum"]),
    )

    output_path = checkpoint_dir / f"{args.name}.pth"
    log_path = log_dir / f"{args.name}.csv"
    write_log_header(log_path)

    print(f"dataset: {args.name}")
    print(f"dataset_root: {args.dataset_root}")
    print(f"device: {device}")
    print(f"num_classes: {len(label_to_name)}")

    best_test_acc = 0.0
    for epoch in range(1, int(config["epochs"]) + 1):
        train_loss, train_acc, _ = run_one_epoch(model, train_loader, criterion, device, optimizer)
        test_loss, test_acc, test_top5_acc = run_one_epoch(model, test_loader, criterion, device)

        append_log(
            log_path,
            [
                epoch,
                round(train_loss, 6),
                round(train_acc, 6),
                round(test_loss, 6),
                round(test_acc, 6),
                round(test_top5_acc, 6),
            ],
        )

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            save_checkpoint(output_path, model, label_to_name, config, epoch, best_test_acc)

        print(
            f"epoch {epoch:03d}/{int(config['epochs'])} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} "
            f"test_top5_acc={test_top5_acc:.4f}"
        )

    print(f"best checkpoint saved to: {output_path}")
    print(f"log saved to: {log_path}")


if __name__ == "__main__":
    main()
