"""Plot DP utility and attack-accuracy curves from pre-computed CSVs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


STYLES = {
    "DP-Pix-b2":     {"marker": "o", "linestyle": "-",  "color": "#0D47A1"},
    "DP-Pix-b4":     {"marker": "o", "linestyle": "-",  "color": "#1976D2"},
    "DP-Pix-b8":     {"marker": "s", "linestyle": "-",  "color": "#42A5F5"},
    "DP-Pix-b16":    {"marker": "s", "linestyle": "--", "color": "#90CAF9"},
    "LP-Blur":       {"marker": "^", "linestyle": "-.", "color": "#FF7043"},
    "DP-Blur-Split": {"marker": "v", "linestyle": ":",  "color": "#D32F2F"},
}

ATTACK_DATASET_TO_METHOD = {
    "dp_pix_b2":         "DP-Pix-b2",
    "dp_pix_b4":         "DP-Pix-b4",
    "dp_pix_b8":         "DP-Pix-b8",
    "dp_pix_b16":        "DP-Pix-b16",
    "lp_blur_k45":       "LP-Blur",
    "dp_blur_split_k45": "DP-Blur-Split",
}


def parse_attack_name(name: str) -> tuple[str | None, float | None]:
    if "_eps" not in name:
        return None, None
    head, eps_token = name.rsplit("_eps", 1)
    try:
        return ATTACK_DATASET_TO_METHOD.get(head), float(eps_token.replace("_", "."))
    except ValueError:
        return None, None


def load_metrics(path: Path) -> dict[str, dict[float, dict[str, float]]]:
    out: dict[str, dict[float, dict[str, float]]] = defaultdict(dict)
    with path.open() as f:
        for row in csv.DictReader(f):
            method = row["method"]
            if method not in STYLES:
                continue
            # Blur mechanisms: only k=45 main sweep. DP-Pix has blank k.
            k = row.get("k") or ""
            if method in {"LP-Blur", "DP-Blur-Split"} and k != "45":
                continue
            try:
                eps = float(row["epsilon"])
                out[method][eps] = {"MSE": float(row["MSE"]), "SSIM": float(row["SSIM"])}
            except ValueError:
                continue
    return out


def load_attack(path: Path) -> dict[str, dict[float, float]]:
    out: dict[str, dict[float, float]] = defaultdict(dict)
    if not path.exists():
        return out
    with path.open() as f:
        for row in csv.DictReader(f):
            method, eps = parse_attack_name(row["dataset"])
            if method is None or eps is None:
                continue
            try:
                out[method][eps] = float(row["top1_acc"])
            except (KeyError, ValueError):
                continue
    return out


def plot_metrics(metrics, output_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for method in STYLES:
        pairs = sorted(metrics.get(method, {}).items())
        if not pairs:
            continue
        eps = [e for e, _ in pairs]
        ax1.plot(eps, [d["MSE"] for _, d in pairs], label=method, **STYLES[method])
        ax2.plot(eps, [d["SSIM"] for _, d in pairs], label=method, **STYLES[method])

    for ax, title, ylabel, yscale, ylim in [
        (ax1, "MSE vs ε", "MSE", "log", None),
        (ax2, "SSIM vs ε", "SSIM", "linear", (0, 1)),
    ]:
        ax.set_xlabel("Privacy Budget ε")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xscale("log")
        if yscale == "log":
            ax.set_yscale("log")
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"saved: {output_path}")


def plot_attack(attack, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in STYLES:
        pairs = sorted(attack.get(method, {}).items())
        if not pairs:
            continue
        ax.plot([e for e, _ in pairs], [a for _, a in pairs], label=method, **STYLES[method])

    ax.axhline(1 / 40, color="black", linestyle=":", linewidth=1, label="Random (1/40)")
    ax.set_xlabel("Privacy Budget ε")
    ax.set_ylabel("Top-1 Attack Accuracy")
    ax.set_title("CNN re-identification attack accuracy vs ε")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", default="data/dp/metrics.csv")
    parser.add_argument("--attack-csv", default="reports/dp_evaluation.csv")
    parser.add_argument("--metrics-output", default="figures/dp_metrics_curves.png")
    parser.add_argument("--attack-output", default="figures/dp_attack_accuracy.png")
    parser.add_argument("--metrics", dest="metrics_csv")
    parser.add_argument("--attack", dest="attack_csv")
    args = parser.parse_args()

    metrics_path = Path(args.metrics_csv)
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics csv not found: {metrics_path}")

    plot_metrics(load_metrics(metrics_path), Path(args.metrics_output))

    attack = load_attack(Path(args.attack_csv))
    if attack:
        plot_attack(attack, Path(args.attack_output))
    else:
        print(f"no attack CSV at {args.attack_csv}, skipping attack plot")


if __name__ == "__main__":
    main()
