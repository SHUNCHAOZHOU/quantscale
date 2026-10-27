"""Publication-style descriptive plot of completed pilot runs, no inference tests."""
import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/quantscale-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot(root):
    root = Path(root)
    if not (root / "completed.json").exists():
        raise ValueError("Pilot has not completed")
    frame = pd.read_csv(root / "comparison.csv")
    order = ["single", "independent", "centralized", "decentralized", "hybrid"]
    means = frame.groupby("topology")[["validation_sharpe", "hidden_sharpe"]].mean().loc[order]
    positions = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(11, 5.6), dpi=180)
    width = 0.33
    validation = ax.bar(positions-width/2, means.validation_sharpe, width, color="#a9b0b8", label="Validation mean (2022-2023)")
    hidden = ax.bar(positions+width/2, means.hidden_sharpe, width, color="#3478b8", label="Held-out mean (2024-2025)")
    for seed, marker, offset in ((7, "o", -0.035), (17, "D", 0.035)):
        values = frame[frame.seed == seed].set_index("topology").loc[order].hidden_sharpe
        ax.scatter(positions+width/2+offset, values, marker=marker, s=40, facecolor="white",
                   edgecolor="#182a3a", linewidth=1.2, zorder=4, label="Held-out repeat %d" % seed)
    ax.bar_label(validation, fmt="%.2f", padding=5, fontsize=10)
    for position, topology, value in zip(positions, order, means.hidden_sharpe):
        lowest = frame[frame.topology == topology].hidden_sharpe.min()
        label_y = min(value, lowest) - 0.14 if value < 0 else value + 0.06
        ax.text(position + width/2, label_y, "%.2f" % value, ha="center",
                va="top" if value < 0 else "bottom", fontsize=10)
    ax.axhline(0, color="#7a8189", linewidth=0.8)
    ax.set_xticks(positions, ["SAS\n1 researcher", "Independent\n4 researchers", "Centralized\n4 + manager", "Decentralized\n4 researchers", "Hybrid\n4 + manager"])
    ax.set_xlabel("Research topology and worker count", labelpad=12)
    ax.set_ylabel("Annualized net Sharpe (zero cash benchmark)")
    ax.set_title("QuantScale: real-price topology pilot", loc="left", fontsize=16, pad=18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e8eb", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.margins(y=0.25)
    ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=9)
    fig.text(0.08, 0.02, "DeepSeek Flash | 12 candidates per run | 2 repeats per topology | fees and borrow costs included\n"
             "Current S&P constituent snapshot; historical holdout. Descriptive pilot, not evidence of statistical superiority.",
             fontsize=8, color="#555b63")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(root / "topology-comparison.png", bbox_inches="tight")
    fig.savefig(root / "topology-comparison.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    plot(parser.parse_args().root)
