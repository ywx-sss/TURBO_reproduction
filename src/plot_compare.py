"""Bar chart: image-only vs with structured table (ST) accuracy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def plot_compare_chart(
    stats: dict[str, Any],
    out_path: Path,
    *,
    title: str = "Impact of structured tables (ST) — TABMWP",
) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError("pip install matplotlib (included in requirements-vl.txt)") from e

    labels = []
    accs = []
    colors = []

    if "with_st" in stats and stats["with_st"].get("accuracy") is not None:
        labels.append("with ST")
        accs.append(stats["with_st"]["accuracy"] * 100)
        colors.append("#4C78A8")
    if "image_only" in stats and stats["image_only"].get("accuracy") is not None:
        labels.append("w/o ST")
        accs.append(stats["image_only"]["accuracy"] * 100)
        colors.append("#F58518")

    if not labels:
        raise ValueError(
            "No accuracy data to plot — all samples failed. "
            "Check compare_results.jsonl errors; fix transformers/torch versions first."
        )

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
    x = range(len(labels))
    bars = ax.bar(x, accs, color=colors, width=0.55, edgecolor="white", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(0, min(100, max(accs) * 1.25 + 5))
    ax.set_title(title, fontsize=12, pad=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    for bar, val in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    meta = []
    if stats.get("model_id"):
        meta.append(str(stats["model_id"]))
    if stats.get("sample_count"):
        meta.append(f"n={stats['sample_count']}")
    if stats.get("delta_with_st_minus_image_only") is not None:
        d = stats["delta_with_st_minus_image_only"] * 100
        meta.append(f"Δ={d:+.1f}pp")
    if meta:
        ax.text(0.5, -0.14, " · ".join(meta), transform=ax.transAxes, ha="center", fontsize=9, color="#555")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def plot_from_stats_file(stats_path: Path, out_path: Path | None = None) -> Path:
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if out_path is None:
        out_path = stats_path.parent.parent.parent / "reports" / "compare_st_vs_image.png"
    return plot_compare_chart(stats, out_path)
