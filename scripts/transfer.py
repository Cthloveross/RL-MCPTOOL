#!/usr/bin/env python3
"""
MCPoisoner Transfer Evaluation
================================
Test attack transferability on an unseen victim model (Llama-3.1-8B).

Usage:
    python scripts/transfer.py --config configs/mve.yaml
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from tabulate import tabulate

from mcpoisoner.baselines import rl_attacker, template_attacker
from mcpoisoner.judge import compute_reward
from mcpoisoner.models import load_attacker_model, load_victim_model
from mcpoisoner.scenarios import AttackScenario, load_scenarios
from mcpoisoner.utils import get_device, gpu_memory_summary, load_config, set_seed, setup_logging


def asr(rewards):
    if not rewards:
        return 0.0
    return sum(1 for r in rewards if r > 0) / len(rewards) * 100


def main():
    parser = argparse.ArgumentParser(description="MCPoisoner Transfer Evaluation")
    parser.add_argument("--config", type=str, default="configs/mve.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed or cfg["experiment"].get("seed", 42)
    set_seed(seed)

    output_dir = cfg["experiment"]["output_dir"]
    checkpoint = args.checkpoint or os.path.join(output_dir, "checkpoints")
    logger = setup_logging(output_dir, "transfer")

    logger.info("=" * 60)
    logger.info("MCPoisoner MVE — Transfer Evaluation")
    logger.info("=" * 60)

    transfer_model_name = cfg["transfer_victim"]["model_name"]
    logger.info("Transfer victim: %s", transfer_model_name)

    # ── Load transfer victim ─────────────────────────────────────
    transfer_model, transfer_tokenizer = load_victim_model(cfg, model_key="transfer_victim")
    device = get_device()

    # ── Load RL attacker ─────────────────────────────────────────
    rl_model, rl_tokenizer = load_attacker_model(cfg, checkpoint_path=checkpoint)
    rl_model = rl_model.to(device)
    logger.info("Models loaded. %s", gpu_memory_summary())

    # ── Load scenarios ───────────────────────────────────────────
    scenarios = load_scenarios(cfg["data"]["scenarios_path"])
    reward_cfg = cfg.get("reward", {})
    eval_cfg = cfg.get("evaluation", {})
    n_variants = eval_cfg.get("num_variants", 4)

    # ── Evaluate ─────────────────────────────────────────────────
    all_results = {}
    for name, attacker_fn in [
        ("Template", lambda s: template_attacker(s)),
        (
            "MCPoisoner-RL",
            lambda s: rl_attacker(s, rl_model, rl_tokenizer, n_variants=n_variants),
        ),
    ]:
        logger.info("\nEvaluating %s on %s...", name, transfer_model_name)
        results = {
            "overall": [], "hijacking": [], "implicit": [],
            "argument_tampering": [], "cross_server": [],
        }
        for scenario in scenarios:
            variants = attacker_fn(scenario)
            best_reward = 0.0
            for desc in variants:
                reward = compute_reward(
                    scenario, desc, transfer_model, transfer_tokenizer, reward_cfg
                )
                best_reward = max(best_reward, reward)
            results["overall"].append(best_reward)
            results[scenario.category].append(best_reward)
            logger.info("  [%s] %s: %.2f", name, scenario.id, best_reward)

        all_results[name] = results

    # ── Print results ────────────────────────────────────────────
    print("\n")
    print("=" * 70)
    print(f"Transfer Results (Victim: {transfer_model_name})")
    print("=" * 70)

    table_data = []
    for name in ["Template", "MCPoisoner-RL"]:
        r = all_results[name]
        table_data.append([
            name,
            f"{asr(r['overall']):.1f}%",
            f"{asr(r['hijacking'] + r['cross_server']):.1f}%",
            f"{asr(r['implicit'] + r['argument_tampering']):.1f}%",
        ])

    print(tabulate(
        table_data,
        headers=["Attacker", "Overall ASR", "Direct ASR", "Implicit ASR"],
        tablefmt="grid",
        colalign=("left", "right", "right", "right"),
    ))

    # ── Save ─────────────────────────────────────────────────────
    results_path = os.path.join(output_dir, "transfer_results.json")
    serializable = {
        name: {k: [float(v) for v in vals] for k, vals in r.items()}
        for name, r in all_results.items()
    }
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main()
