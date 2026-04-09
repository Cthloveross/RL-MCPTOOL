#!/usr/bin/env python3
"""
MCPoisoner GRPO Training Script
================================
Train the RL attacker using Group Relative Policy Optimization.

Usage:
    python scripts/train.py --config configs/mve.yaml
    python scripts/train.py --config configs/mve.yaml --seed 123
"""

import argparse
import os
import sys

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from mcpoisoner.models import get_lora_config, load_victim_model
from mcpoisoner.prompts import build_attacker_messages
from mcpoisoner.reward import MCPRewardFunction
from mcpoisoner.scenarios import load_scenarios
from mcpoisoner.utils import gpu_memory_summary, load_config, set_seed, setup_logging


def build_training_dataset(scenarios, tokenizer) -> Dataset:
    """Build HuggingFace Dataset for GRPOTrainer.

    Each entry is a list of chat messages (system + user) that the
    GRPOTrainer will format using the tokenizer's chat template.
    """
    records = []
    for scenario in scenarios:
        messages = build_attacker_messages(scenario)
        records.append({"prompt": messages})
    return Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser(description="MCPoisoner GRPO Training")
    parser.add_argument(
        "--config", type=str, default="configs/mve.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override config seed")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed or cfg["experiment"].get("seed", 42)
    set_seed(seed)

    output_dir = cfg["experiment"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir, "train")

    logger.info("=" * 60)
    logger.info("MCPoisoner MVE — GRPO Training")
    logger.info("=" * 60)
    logger.info("Config: %s", args.config)
    logger.info("Output: %s", output_dir)
    logger.info("Seed: %d", seed)

    # ── Load victim model (stays on GPU throughout) ──────────────
    victim_model, victim_tokenizer = load_victim_model(cfg, model_key="victim")
    logger.info("Victim loaded. %s", gpu_memory_summary())

    # ── Load attacker tokenizer (model is loaded by GRPOTrainer) ─
    from transformers import AutoTokenizer
    attacker_name = cfg["attacker"]["model_name"]
    attacker_tokenizer = AutoTokenizer.from_pretrained(
        attacker_name, trust_remote_code=True
    )
    if attacker_tokenizer.pad_token is None:
        attacker_tokenizer.pad_token = attacker_tokenizer.eos_token
    attacker_tokenizer.padding_side = "left"

    # ── Load scenarios & build dataset ───────────────────────────
    scenarios = load_scenarios(cfg["data"]["scenarios_path"])
    dataset = build_training_dataset(scenarios, attacker_tokenizer)
    logger.info("Training dataset: %d scenarios", len(dataset))

    # ── Reward function ──────────────────────────────────────────
    reward_cfg = cfg.get("reward", {})
    reward_fn = MCPRewardFunction(scenarios, victim_model, victim_tokenizer, reward_cfg)

    # ── LoRA config ──────────────────────────────────────────────
    lora_config = get_lora_config(cfg)

    # ── GRPO config ──────────────────────────────────────────────
    train_cfg = cfg["training"]
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    grpo_config = GRPOConfig(
        output_dir=checkpoint_dir,
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 2),
        num_generations=train_cfg["num_generations"],
        max_completion_length=train_cfg["max_completion_length"],
        max_prompt_length=train_cfg["max_prompt_length"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        beta=train_cfg.get("beta", 0.0),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        bf16=train_cfg.get("bf16", True),
        save_steps=train_cfg.get("save_steps", 200),
        logging_steps=train_cfg.get("logging_steps", 10),
        log_completions=train_cfg.get("log_completions", True),
        report_to=train_cfg.get("report_to", "none"),
        remove_unused_columns=False,
        seed=seed,
    )

    # ── Create trainer ───────────────────────────────────────────
    trainer = GRPOTrainer(
        model=attacker_name,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=attacker_tokenizer,
    )

    logger.info("Training config:")
    logger.info("  Attacker: %s + LoRA (r=%d)", attacker_name, cfg["attacker"]["lora"]["rank"])
    logger.info("  Victim: %s (4-bit)", cfg["victim"]["model_name"])
    logger.info("  Group size G=%d", train_cfg["num_generations"])
    logger.info("  Epochs: %d", train_cfg["num_train_epochs"])
    logger.info("  Batch size: %d", train_cfg["per_device_train_batch_size"])
    logger.info("  LR: %s", train_cfg["learning_rate"])
    logger.info("  %s", gpu_memory_summary())

    # ── Train ────────────────────────────────────────────────────
    trainer.train()
    trainer.save_model(checkpoint_dir)
    logger.info("Training complete. Checkpoint saved to %s", checkpoint_dir)


if __name__ == "__main__":
    main()
