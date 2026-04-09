#!/usr/bin/env python3
"""
MCPoisoner Evaluation Script
==============================
Evaluate all baselines and the RL attacker on 20 MVE scenarios.

Usage:
    python scripts/evaluate.py --config configs/mve.yaml
    python scripts/evaluate.py --config configs/mve.yaml --checkpoint ./experiments/mve/checkpoints
"""

import argparse
import json
import os
import sys
from typing import Callable, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from tabulate import tabulate

from mcpoisoner.baselines import (
    llm_singleshot_attacker,
    random_attacker,
    rl_attacker,
    template_attacker,
)
from mcpoisoner.judge import compute_reward
from mcpoisoner.models import load_attacker_model, load_victim_model
from mcpoisoner.scenarios import AttackScenario, load_scenarios
from mcpoisoner.utils import get_device, gpu_memory_summary, load_config, set_seed, setup_logging


def evaluate_attacker(
    name: str,
    attacker_fn: Callable,
    scenarios: List[AttackScenario],
    victim_model,
    victim_tokenizer,
    reward_cfg: dict,
    logger,
) -> Dict[str, List[float]]:
    """Run a single attacker across all scenarios, return per-category rewards."""
    results: Dict[str, List[float]] = {
        "overall": [],
        "hijacking": [],
        "implicit": [],
        "argument_tampering": [],
        "cross_server": [],
    }

    for i, scenario in enumerate(scenarios):
        variants = attacker_fn(scenario)
        best_reward = 0.0
        for desc in variants:
            reward = compute_reward(
                scenario, desc, victim_model, victim_tokenizer, reward_cfg
            )
            best_reward = max(best_reward, reward)

        results["overall"].append(best_reward)
        results[scenario.category].append(best_reward)
        logger.info(
            "  [%s] %s (%s): best_reward=%.2f (%d variants)",
            name, scenario.id, scenario.category, best_reward, len(variants),
        )

    return results


def asr(rewards: List[float]) -> float:
    """ASR = fraction of nonzero rewards."""
    if not rewards:
        return 0.0
    return sum(1 for r in rewards if r > 0) / len(rewards) * 100


def main():
    parser = argparse.ArgumentParser(description="MCPoisoner Evaluation")
    parser.add_argument("--config", type=str, default="configs/mve.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="RL checkpoint path (default: <output_dir>/checkpoints)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed or cfg["experiment"].get("seed", 42)
    set_seed(seed)

    output_dir = cfg["experiment"]["output_dir"]
    checkpoint = args.checkpoint or os.path.join(output_dir, "checkpoints")
    logger = setup_logging(output_dir, "evaluate")

    logger.info("=" * 60)
    logger.info("MCPoisoner MVE — Evaluation")
    logger.info("=" * 60)

    # ── Load models ──────────────────────────────────────────────
    victim_model, victim_tokenizer = load_victim_model(cfg, model_key="victim")
    device = get_device()

    # Base attacker (for LLM-SingleShot)
    base_model, base_tokenizer = load_attacker_model(cfg, checkpoint_path=None)
    base_model = base_model.to(device)

    # RL attacker
    rl_model, rl_tokenizer = load_attacker_model(cfg, checkpoint_path=checkpoint)
    rl_model = rl_model.to(device)

    logger.info("Models loaded. %s", gpu_memory_summary())

    # ── Load scenarios ───────────────────────────────────────────
    scenarios = load_scenarios(cfg["data"]["scenarios_path"])
    reward_cfg = cfg.get("reward", {})
    eval_cfg = cfg.get("evaluation", {})
    n_variants = eval_cfg.get("num_variants", 4)
    temperature = eval_cfg.get("temperature", 0.7)
    top_p = eval_cfg.get("top_p", 0.9)

    # ── Define attackers ─────────────────────────────────────────
    attackers = [
        ("Random", lambda s: random_attacker(s, n_variants=n_variants)),
        ("Template", lambda s: template_attacker(s)),
        (
            "LLM-SingleShot",
            lambda s: llm_singleshot_attacker(
                s, base_model, base_tokenizer,
                n_variants=n_variants, temperature=temperature, top_p=top_p,
            ),
        ),
        (
            "MCPoisoner-RL",
            lambda s: rl_attacker(
                s, rl_model, rl_tokenizer,
                n_variants=n_variants, temperature=temperature, top_p=top_p,
            ),
        ),
    ]

    # ── Evaluate ─────────────────────────────────────────────────
    all_results = {}
    for name, fn in attackers:
        logger.info("\nEvaluating %s...", name)
        results = evaluate_attacker(
            name, fn, scenarios, victim_model, victim_tokenizer, reward_cfg, logger
        )
        all_results[name] = results

    # ── Print main results table ─────────────────────────────────
    print("\n")
    print("=" * 70)
    print("MCPoisoner MVE Results")
    print("=" * 70)

    table_data = []
    for name in ["Random", "Template", "LLM-SingleShot", "MCPoisoner-RL"]:
        r = all_results[name]
        overall = asr(r["overall"])
        direct = asr(r["hijacking"] + r["cross_server"])
        implicit = asr(r["implicit"] + r["argument_tampering"])
        table_data.append([name, f"{overall:.1f}%", f"{direct:.1f}%", f"{implicit:.1f}%"])

    print(tabulate(
        table_data,
        headers=["Attacker", "Overall ASR", "Direct ASR", "Implicit ASR"],
        tablefmt="grid",
        colalign=("left", "right", "right", "right"),
    ))

    # ── Print per-category breakdown ─────────────────────────────
    print("\nPer-Category Breakdown:")
    cat_data = []
    for name in ["Random", "Template", "LLM-SingleShot", "MCPoisoner-RL"]:
        r = all_results[name]
        cat_data.append([
            name,
            f"{asr(r['hijacking']):.1f}%",
            f"{asr(r['implicit']):.1f}%",
            f"{asr(r['argument_tampering']):.1f}%",
            f"{asr(r['cross_server']):.1f}%",
        ])

    print(tabulate(
        cat_data,
        headers=["Attacker", "Hijacking", "Implicit", "Arg Tampering", "Cross-Server"],
        tablefmt="grid",
        colalign=("left", "right", "right", "right", "right"),
    ))

    # ── Save results ─────────────────────────────────────────────
    results_path = os.path.join(output_dir, "evaluation_results.json")
    serializable = {
        name: {k: [float(v) for v in vals] for k, vals in r.items()}
        for name, r in all_results.items()
    }
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main()
