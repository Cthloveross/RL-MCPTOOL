#!/usr/bin/env python3
"""MCPAlign DPO baseline training.

Trains the agent with offline DPO using the same data as GRPO.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datasets import Dataset
from transformers import AutoTokenizer
from trl import DPOConfig, DPOTrainer

from mcpalign.models import get_lora_config
from mcpalign.utils import load_config, set_seed, setup_logging


def main():
    parser = argparse.ArgumentParser(description="MCPAlign DPO Baseline Training")
    parser.add_argument("--config", type=str, default="configs/mcpalign_mve.yaml")
    parser.add_argument("--sft-checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed or cfg["experiment"].get("seed", 42)
    set_seed(seed)

    output_dir = cfg["experiment"]["output_dir"]
    dpo_dir = os.path.join(output_dir, "dpo_checkpoint")
    sft_checkpoint = args.sft_checkpoint or os.path.join(output_dir, "sft_checkpoint")
    os.makedirs(dpo_dir, exist_ok=True)
    logger = setup_logging(output_dir, "train_dpo")

    logger.info("=" * 60)
    logger.info("MCPAlign — DPO Baseline Training")
    logger.info("=" * 60)

    # Load DPO data
    dpo_path = cfg["data"]["dpo_data_path"]
    with open(dpo_path) as f:
        dpo_data = json.load(f)
    logger.info("Loaded %d DPO pairs from %s", len(dpo_data), dpo_path)

    # Build HF dataset
    dataset = Dataset.from_list([{
        "prompt": d["prompt"],
        "chosen": d["chosen"],
        "rejected": d["rejected"],
    } for d in dpo_data])

    # Tokenizer
    model_name = cfg["agent"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA config
    lora_config = get_lora_config(cfg)

    # DPO config
    dpo_cfg = cfg.get("dpo", {})
    training_args = DPOConfig(
        output_dir=dpo_dir,
        num_train_epochs=dpo_cfg.get("num_epochs", 1),
        per_device_train_batch_size=dpo_cfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=dpo_cfg.get("gradient_accumulation_steps", 2),
        learning_rate=dpo_cfg.get("learning_rate", 5e-7),
        beta=dpo_cfg.get("beta", 0.1),
        max_length=dpo_cfg.get("max_length", 2048),
        save_steps=dpo_cfg.get("save_steps", 100),
        logging_steps=dpo_cfg.get("logging_steps", 10),
        bf16=True,
        report_to=dpo_cfg.get("report_to", "none"),
        seed=seed,
    )

    # Use SFT checkpoint as starting point
    model_to_load = sft_checkpoint if os.path.exists(sft_checkpoint) else model_name
    logger.info("Loading model from: %s", model_to_load)

    trainer = DPOTrainer(
        model=model_to_load,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    logger.info("DPO Config: beta=%.2f, lr=%s, epochs=%d, pairs=%d",
                dpo_cfg.get("beta", 0.1), dpo_cfg.get("learning_rate", 5e-7),
                dpo_cfg.get("num_epochs", 1), len(dataset))

    trainer.train()
    trainer.save_model(dpo_dir)
    logger.info("DPO training complete. Checkpoint saved to %s", dpo_dir)


if __name__ == "__main__":
    main()
