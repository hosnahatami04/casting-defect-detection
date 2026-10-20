"""Figure rendering for evaluation artifacts. Matplotlib only; every number
comes from a JSON file produced by an earlier command."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _use_agg() -> None:
    import matplotlib

    matplotlib.use("Agg")


def render_efficiency_curve(efficiency_json: Path, output: Path) -> None:
    _use_agg()
    import matplotlib.pyplot as plt

    data = json.loads(efficiency_json.read_text(encoding="utf-8"))
    by_fraction: dict[float, list[float]] = defaultdict(list)
    defects: dict[float, list[int]] = defaultdict(list)
    for point in data["points"]:
        by_fraction[point["fraction"]].append(point["auroc"])
        defects[point["fraction"]].append(point["n_defects_train"])
    fractions = sorted(by_fraction)
    means = [float(np.mean(by_fraction[f])) for f in fractions]
    lows = [float(np.min(by_fraction[f])) for f in fractions]
    highs = [float(np.max(by_fraction[f])) for f in fractions]
    counts = [int(np.mean(defects[f])) for f in fractions]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(counts, means, "o-", color="#1f77b4", label="EfficientNet (supervised)")
    ax.fill_between(counts, lows, highs, color="#1f77b4", alpha=0.2, label="min/max over 3 seeds")
    oneclass = data["oneclass_knn"]["auroc"]
    ax.axhline(
        oneclass["point"],
        color="#d62728",
        linestyle="--",
        label=f"one-class kNN, 0 defects ({oneclass['point']:.3f})",
    )
    ax.set_xlabel("defective training images")
    ax.set_ylabel("out-of-fold AUROC")
    ax.set_title("Data efficiency: AUROC vs defect count")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120)
    plt.close(fig)


def render_robustness_bars(robustness_json: Path, output: Path) -> None:
    _use_agg()
    import matplotlib.pyplot as plt

    data = json.loads(robustness_json.read_text(encoding="utf-8"))
    rows = data["rows"]
    baseline = next(r["defect_recall"] for r in rows if r["perturbation"] == "none")
    perturbations = [r["perturbation"] for r in rows if r["perturbation"] != "none"]
    names = sorted(set(perturbations))
    drops: dict[str, list[float]] = {name: [] for name in names}
    for row in rows:
        if row["perturbation"] == "none":
            continue
        drops[row["perturbation"]].append(100.0 * (baseline - row["defect_recall"]))

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(names))
    width = 0.25
    for level in range(3):
        values = [drops[name][level] for name in names]
        ax.bar(x + (level - 1) * width, values, width, label=f"severity {level + 1}")
    ax.axhline(3.0, color="#555", linestyle=":", label="3-point target")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("defect-recall drop (points)")
    ax.set_title("Robustness: recall drop under perturbation (OOF)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120)
    plt.close(fig)


def render_reliability(calibration_json: Path, output: Path) -> None:
    _use_agg()
    import matplotlib.pyplot as plt

    data = json.loads(calibration_json.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
    for key, color, label in (
        ("reliability_before", "#d62728", f"before (ECE {data['ece_before']:.4f})"),
        ("reliability_after", "#1f77b4", f"after (ECE {data['ece_after']:.4f})"),
    ):
        bins = data[key]
        ax.plot(
            [b["mean_probability"] for b in bins],
            [b["fraction_defective"] for b in bins],
            "o-",
            color=color,
            label=label,
        )
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("fraction defective")
    ax.set_title(f"Reliability (temperature {data['temperature']:.3f})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120)
    plt.close(fig)
