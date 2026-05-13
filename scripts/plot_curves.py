"""Plot DP utility curves (MSE/SSIM vs epsilon) and DP attack accuracy.

Reads pre-computed CSVs — does NOT regenerate any DP images:
  * --metrics-csv  (data/dp/metrics.csv)  : method, k, epsilon, MSE, SSIM, n
  * --attack-csv   (reports/dp_evaluation.csv) : dataset, top1_acc, ...

Outputs:
  * figures/dp_metrics_curves.png    — MSE and SSIM vs ε per method
  * figures/dp_attack_accuracy.png   — Top-1 attack accuracy vs ε per method
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


# Method display order, marker, line style, and color
STYLES = {
    "DP-Pix-b2":      {"marker": "o", "linestyle": "-",  "color": "#0D47A1"},
    "DP-Pix-b4":      {"marker": "o", "linestyle": "-",  "color": "#1976D2"},
    "DP-Pix-b8":      {"marker": "s", "linestyle": "-",  "color": "#42A5F5"},
    "DP-Pix-b16":     {"marker": "s", "linestyle": "--", "color": "#90CAF9"},
    "LP-Blur":        {"marker": "^", "linestyle": "-.", "color": "#FF7043"},
    "DP-Blur-Split":  {"marker": "v", "linestyle": ":",  "color": "#D32F2F"},
}

# Map attack CSV dataset names → method labels used in the metrics CSV
ATTACK_DATASET_TO_METHOD = {
    "dp_pix_b2":               "DP-Pix-b2",
    "dp_pix_b4":               "DP-Pix-b4",
    "dp_pix_b8":               "DP-Pix-b8",
    "dp_pix_b16":              "DP-Pix-b16",
    "lp_blur_k45":             "LP-Blur",
    "dp_blur_split_k45":       "DP-Blur-Split",
}


def parse_attack_name(dataset_name: str) -> tuple[str | None, float | None]:
    """Parse ``dp_pix_b8_eps0_1`` → (DP-Pix-b8, 0.1) and similar."""
    if "_eps" not in dataset_name:
        return None, None
    head, eps_token = dataset_name.rsplit("_eps", 1)
    try:
        eps = float(eps_token.replace("_", "."))
    except ValueError:
        return None, None
    method = ATTACK_DATASET_TO_METHOD.get(head)
    return method, eps


def load_metrics(path: Path) -> dict[str, dict[float, dict[str, float]]]:
    """Return ``{method: {epsilon: {"MSE": x, "SSIM": y}}}`` keyed by method."""
    out: dict[str, dict[float, dict[str, float]]] = defaultdict(dict)
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["method"]
            if method not in STYLES:
                continue
            # Only keep the main ε-scan rows (LP-Blur/DP-Blur-Split have k=45;
            # DP-Pix has no k column populated — accept blank). Visual-only
            # k=15/99 rows are skipped because they have only one ε value.
            k = row.get("k") or ""
            if method in {"LP-Blur", "DP-Blur-Split"} and k != "45":
                continue
            try:
                eps = float(row["epsilon"])
                out[method][eps] = {
                    "MSE": float(row["MSE"]),
                    "SSIM": float(row["SSIM"]),
                }
            except ValueError:
                continue
    return out


def load_attack(path: Path) -> dict[str, dict[float, float]]:
    """Return ``{method: {epsilon: top1_acc}}``."""
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


def plot_metrics(
    metrics: dict[str, dict[float, dict[str, float]]],
    output_path: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for method in STYLES:
        if method not in metrics:
            continue
        eps_pairs = sorted(metrics[method].items())
        if not eps_pairs:
            continue
        epsilons = [e for e, _ in eps_pairs]
        mses = [d["MSE"] for _, d in eps_pairs]
        ssims = [d["SSIM"] for _, d in eps_pairs]
        ax1.plot(epsilons, mses, label=method, **STYLES[method])
        ax2.plot(epsilons, ssims, label=method, **STYLES[method])

    ax1.set_xlabel("Privacy Budget ε")
    ax1.set_ylabel("MSE")
    ax1.set_title("MSE vs ε")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc="best")

    ax2.set_xlabel("Privacy Budget ε")
    ax2.set_ylabel("SSIM")
    ax2.set_title("SSIM vs ε")
    ax2.set_xscale("log")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc="best")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"saved: {output_path}")


def plot_attack(
    attack: dict[str, dict[float, float]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    plotted = 0
    for method in STYLES:
        if method not in attack:
            continue
        eps_pairs = sorted(attack[method].items())
        if not eps_pairs:
            continue
        epsilons = [e for e, _ in eps_pairs]
        accs = [a for _, a in eps_pairs]
        ax.plot(epsilons, accs, label=method, **STYLES[method])
        plotted += 1

    ax.axhline(1 / 40, color="black", linestyle=":", linewidth=1, label="Random (1/40)")
    ax.set_xlabel("Privacy Budget ε")
    ax.set_ylabel("Top-1 Attack Accuracy")
    ax.set_title("CNN re-identification attack accuracy vs ε")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"saved: {output_path}  ({plotted} methods plotted)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", default="data/dp/metrics.csv")
    parser.add_argument("--attack-csv", default="reports/dp_evaluation.csv")
    parser.add_argument(
        "--metrics-output",
        default="figures/dp_metrics_curves.png",
    )
    parser.add_argument(
        "--attack-output",
        default="figures/dp_attack_accuracy.png",
    )
    # Backwards-compatible aliases for the names suggested in docs/RESUME.md
    parser.add_argument("--metrics", dest="metrics_csv", help="Alias for --metrics-csv")
    parser.add_argument("--attack", dest="attack_csv", help="Alias for --attack-csv")
    args = parser.parse_args()

    metrics_path = Path(args.metrics_csv)
    attack_path = Path(args.attack_csv)

    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics csv not found: {metrics_path}")

    metrics = load_metrics(metrics_path)
    print(f"loaded metrics for {len(metrics)} methods from {metrics_path}")
    plot_metrics(metrics, Path(args.metrics_output))

    attack = load_attack(attack_path)
    if attack:
        print(f"loaded attack accuracy for {len(attack)} methods from {attack_path}")
        plot_attack(attack, Path(args.attack_output))
    else:
        print(f"no attack accuracy at {attack_path}, skipping attack plot")


if __name__ == "__main__":
    main()
