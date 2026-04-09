#!/usr/bin/env python3
"""Generate SFT and DPO training data for MCPAlign."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mcpalign.dpo_data import generate_dpo_pairs, save_dpo_dataset
from mcpalign.environment import MTMCPGym
from mcpalign.sft_data import generate_sft_dataset, save_sft_dataset
from mcpalign.utils import load_config, set_seed, setup_logging


def main():
    parser = argparse.ArgumentParser(description="Generate MCPAlign training data")
    parser.add_argument("--config", type=str, default="configs/mcpalign_mve.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sft-only", action="store_true")
    parser.add_argument("--dpo-only", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed or cfg["experiment"].get("seed", 42)
    set_seed(seed)

    output_dir = cfg["experiment"]["output_dir"]
    logger = setup_logging(output_dir, "generate_data")
    logger.info("Generating MCPAlign training data")

    # Initialize MT-MCPGym
    env_cfg = cfg["environment"]
    gym = MTMCPGym(
        tool_registry_path=cfg["data"]["tool_registry_path"],
        multistep_tasks_path=cfg["data"].get(
            "multistep_tasks_path", "data/mcpalign/multistep_tasks.json"
        ),
        attack_templates_dir=cfg["data"]["attack_templates_dir"],
        benign_ratio=env_cfg.get("benign_ratio", 0.5),
        active_families=env_cfg.get("attack_families", ["P1"]),
    )

    # Generate SFT data
    if not args.dpo_only:
        sft_cfg = cfg.get("sft", {})
        num_sft = sft_cfg.get("num_samples", 500)
        sft_data = generate_sft_dataset(gym, num_samples=num_sft)
        sft_path = cfg["data"]["sft_data_path"]
        os.makedirs(os.path.dirname(sft_path), exist_ok=True)
        save_sft_dataset(sft_data, sft_path)

    # Generate DPO data
    if not args.sft_only:
        dpo_cfg = cfg.get("dpo", {})
        num_dpo = dpo_cfg.get("num_pairs", 500)
        dpo_pairs = generate_dpo_pairs(gym, num_pairs=num_dpo)
        dpo_path = cfg["data"]["dpo_data_path"]
        os.makedirs(os.path.dirname(dpo_path), exist_ok=True)
        save_dpo_dataset(dpo_pairs, dpo_path)

    logger.info("Data generation complete")


if __name__ == "__main__":
    main()
