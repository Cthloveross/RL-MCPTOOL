#!/usr/bin/env python3
"""MCPAlign SFT warm-start training.

Trains the agent to output valid JSON actions and basic safety behaviors.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from mcpalign.models import get_lora_config
from mcpalign.utils import gpu_memory_summary, load_config, set_seed, setup_logging


def main():
    parser = argparse.ArgumentParser(description="MCPAlign SFT Training")
    parser.add_argument("--config", type=str, default="configs/mcpalign_mve.yaml")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed or cfg["experiment"].get("seed", 42)
    set_seed(seed)

    output_dir = cfg["experiment"]["output_dir"]
    sft_dir = os.path.join(output_dir, "sft_checkpoint")
    os.makedirs(sft_dir, exist_ok=True)
    logger = setup_logging(output_dir, "train_sft")

    logger.info("=" * 60)
    logger.info("MCPAlign — SFT Warm-Start Training")
    logger.info("=" * 60)

    # Load SFT data
    sft_path = cfg["data"]["sft_data_path"]
    with open(sft_path) as f:
        sft_data = json.load(f)
    logger.info("Loaded %d SFT examples from %s", len(sft_data), sft_path)

    # Convert to HF dataset
    dataset = Dataset.from_list([{"messages": d["messages"]} for d in sft_data])

    # Model & tokenizer
    model_name = cfg["agent"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA config
    lora_config = get_lora_config(cfg)

    # SFT config
    sft_cfg = cfg.get("sft", {})
    training_args = SFTConfig(
        output_dir=sft_dir,
        num_train_epochs=sft_cfg.get("num_epochs", 3),
        per_device_train_batch_size=sft_cfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=sft_cfg.get("gradient_accumulation_steps", 2),
        learning_rate=sft_cfg.get("learning_rate", 2e-5),
        max_seq_length=sft_cfg.get("max_seq_length", 2048),
        save_steps=sft_cfg.get("save_steps", 100),
        logging_steps=sft_cfg.get("logging_steps", 10),
        bf16=True,
        report_to=sft_cfg.get("report_to", "none"),
        seed=seed,
    )

    # Train
    trainer = SFTTrainer(
        model=model_name,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    logger.info("Model: %s", model_name)
    logger.info("SFT examples: %d", len(dataset))
    logger.info("Epochs: %d", sft_cfg.get("num_epochs", 3))

    trainer.train()
    trainer.save_model(sft_dir)
    logger.info("SFT training complete. Checkpoint saved to %s", sft_dir)


if __name__ == "__main__":
    main()
