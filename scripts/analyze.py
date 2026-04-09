#!/usr/bin/env python3
"""
MCPoisoner Results Analysis & Visualization
=============================================
Generate publication-quality figures and tables from experiment results.

Usage:
    python scripts/analyze.py --config configs/mve.yaml
    python scripts/analyze.py --config configs/mve.yaml --eval-only
    python scripts/analyze.py --config configs/mve.yaml --transfer-only
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for servers
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tabulate import tabulate

from mcpoisoner.utils import load_config


def asr(rewards):
    if not rewards:
        return 0.0
    return sum(1 for r in rewards if r > 0) / len(rewards) * 100


def avg_reward(rewards):
    if not rewards:
        return 0.0
    return sum(rewards) / len(rewards)


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_overall_asr(results: dict, output_path: str, fig_format: str = "pdf", dpi: int = 300):
    """Bar chart: Overall ASR comparison across attackers."""
    attackers = list(results.keys())
    overall_asrs = [asr(results[a]["overall"]) for a in attackers]

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = sns.color_palette("Set2", len(attackers))
    bars = ax.bar(attackers, overall_asrs, color=colors, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, overall_asrs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title("Overall ASR Comparison")
    ax.set_ylim(0, max(overall_asrs) * 1.2 + 5)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, format=fig_format, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_category_heatmap(results: dict, output_path: str, fig_format: str = "pdf", dpi: int = 300):
    """Heatmap: ASR by attacker x category."""
    attackers = list(results.keys())
    categories = ["hijacking", "implicit", "argument_tampering", "cross_server"]
    cat_labels = ["Hijacking", "Implicit", "Arg Tampering", "Cross-Server"]

    matrix = np.zeros((len(attackers), len(categories)))
    for i, a in enumerate(attackers):
        for j, c in enumerate(categories):
            matrix[i, j] = asr(results[a].get(c, []))

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(
        matrix, annot=True, fmt=".1f", cmap="YlOrRd",
        xticklabels=cat_labels, yticklabels=attackers,
        ax=ax, vmin=0, vmax=100,
        cbar_kws={"label": "ASR (%)"},
    )
    ax.set_title("ASR by Attacker and Category")

    plt.tight_layout()
    plt.savefig(output_path, format=fig_format, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_reward_distribution(results: dict, output_path: str, fig_format: str = "pdf", dpi: int = 300):
    """Box plot: reward distribution per attacker."""
    attackers = list(results.keys())

    fig, ax = plt.subplots(figsize=(6, 4))
    data = [results[a]["overall"] for a in attackers]
    bp = ax.boxplot(data, labels=attackers, patch_artist=True, showmeans=True)

    colors = sns.color_palette("Set2", len(attackers))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_ylabel("Reward")
    ax.set_title("Reward Distribution per Attacker")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, format=fig_format, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def generate_latex_table(results: dict, output_path: str):
    """Generate LaTeX table for the paper."""
    attackers = list(results.keys())
    categories = ["hijacking", "implicit", "argument_tampering", "cross_server"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Attacker} & \textbf{Overall} & \textbf{Direct} & \textbf{Implicit} "
        r"& \textbf{Hijack} & \textbf{ArgTamp} & \textbf{XServer} \\",
        r"\midrule",
    ]

    for a in attackers:
        r = results[a]
        overall = asr(r["overall"])
        direct = asr(r.get("hijacking", []) + r.get("cross_server", []))
        implicit = asr(r.get("implicit", []) + r.get("argument_tampering", []))
        hijack = asr(r.get("hijacking", []))
        argtamp = asr(r.get("argument_tampering", []))
        xserver = asr(r.get("cross_server", []))

        # Bold the best values
        vals = [overall, direct, implicit, hijack, argtamp, xserver]
        val_strs = [f"{v:.1f}\\%" for v in vals]
        lines.append(f"{a} & {' & '.join(val_strs)} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{MVE evaluation results: ASR (\%) across attack categories.}",
        r"\label{tab:mve-results}",
        r"\end{table}",
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="MCPoisoner Results Analysis")
    parser.add_argument("--config", type=str, default="configs/mve.yaml")
    parser.add_argument("--eval-only", action="store_true", help="Only analyze evaluation results")
    parser.add_argument("--transfer-only", action="store_true", help="Only analyze transfer results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = cfg["experiment"]["output_dir"]
    analysis_cfg = cfg.get("analysis", {})
    fig_format = analysis_cfg.get("figure_format", "pdf")
    dpi = analysis_cfg.get("dpi", 300)

    style = analysis_cfg.get("plot_style", "seaborn-v0_8-paper")
    try:
        plt.style.use(style)
    except OSError:
        plt.style.use("seaborn-v0_8-paper" if "seaborn" in style else "default")

    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # ── Evaluation results ───────────────────────────────────────
    eval_path = os.path.join(output_dir, "evaluation_results.json")
    if not args.transfer_only and os.path.exists(eval_path):
        print("\n=== Evaluation Results ===")
        results = load_results(eval_path)

        plot_overall_asr(
            results, os.path.join(figures_dir, f"overall_asr.{fig_format}"),
            fig_format=fig_format, dpi=dpi,
        )
        plot_category_heatmap(
            results, os.path.join(figures_dir, f"category_heatmap.{fig_format}"),
            fig_format=fig_format, dpi=dpi,
        )
        plot_reward_distribution(
            results, os.path.join(figures_dir, f"reward_distribution.{fig_format}"),
            fig_format=fig_format, dpi=dpi,
        )
        generate_latex_table(
            results, os.path.join(figures_dir, "results_table.tex"),
        )
    elif not args.transfer_only:
        print(f"Evaluation results not found at {eval_path}. Run evaluate.py first.")

    # ── Transfer results ─────────────────────────────────────────
    transfer_path = os.path.join(output_dir, "transfer_results.json")
    if not args.eval_only and os.path.exists(transfer_path):
        print("\n=== Transfer Results ===")
        transfer_results = load_results(transfer_path)

        plot_overall_asr(
            transfer_results, os.path.join(figures_dir, f"transfer_asr.{fig_format}"),
            fig_format=fig_format, dpi=dpi,
        )
    elif not args.eval_only:
        print(f"Transfer results not found at {transfer_path}. Run transfer.py first.")

    print(f"\nAll figures saved to {figures_dir}/")


if __name__ == "__main__":
    main()
