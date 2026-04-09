#!/usr/bin/env python3
"""MCPAlign results analysis: figures, LaTeX tables, and case studies."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tabulate import tabulate

from mcpalign.utils import load_config


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_asr_comparison(results: dict, output_path: str, dpi: int = 300):
    """Bar chart comparing ASR across defenses."""
    defenses = list(results.keys())
    asrs = [results[d]["ASR"] for d in defenses]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = sns.color_palette("Set2", len(defenses))
    bars = ax.bar(defenses, asrs, color=colors, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, asrs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Attack Success Rate (%) ↓")
    ax.set_title("ASR Comparison: MCPAlign vs Baselines")
    ax.set_ylim(0, max(asrs) * 1.3 + 5)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_security_utility_tradeoff(results: dict, output_path: str, dpi: int = 300):
    """Scatter plot: (1-ASR) vs BTSR — the Pareto frontier."""
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = sns.color_palette("Set2", len(results))

    for i, (name, m) in enumerate(results.items()):
        security = 100 - m["ASR"]
        utility = m["BTSR"]
        ax.scatter(utility, security, s=120, c=[colors[i]], edgecolors="black",
                   linewidths=0.5, zorder=3, label=name)
        ax.annotate(name, (utility, security), textcoords="offset points",
                    xytext=(8, 5), fontsize=8)

    ax.set_xlabel("BTSR (Utility) ↑")
    ax.set_ylabel("1 - ASR (Security) ↑")
    ax.set_title("Security–Utility Trade-off")
    ax.set_xlim(50, 105)
    ax.set_ylim(50, 105)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def generate_latex_table(results: dict, output_path: str):
    """Generate LaTeX table for the paper."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Defense} & \textbf{ASR$\downarrow$} & \textbf{BTSR$\uparrow$} & \textbf{ORR$\downarrow$} \\",
        r"\midrule",
    ]
    for name, m in results.items():
        lines.append(f"{name} & {m['ASR']:.1f}\\% & {m['BTSR']:.1f}\\% & {m['ORR']:.1f}\\% \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{MCPAlign evaluation results on MVE test set.}",
        r"\label{tab:mve-results}",
        r"\end{table}",
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="MCPAlign Results Analysis")
    parser.add_argument("--config", type=str, default="configs/mcpalign_mve.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = cfg["experiment"]["output_dir"]
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    eval_path = os.path.join(output_dir, "evaluation_results.json")
    if not os.path.exists(eval_path):
        print(f"No results at {eval_path}. Run mcpalign_evaluate.py first.")
        return

    results = load_results(eval_path)

    plot_asr_comparison(results, os.path.join(figures_dir, "asr_comparison.pdf"))
    plot_security_utility_tradeoff(results, os.path.join(figures_dir, "pareto_frontier.pdf"))
    generate_latex_table(results, os.path.join(figures_dir, "results_table.tex"))

    print(f"\nAll figures saved to {figures_dir}/")


if __name__ == "__main__":
    main()
