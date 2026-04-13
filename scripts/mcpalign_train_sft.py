#!/usr/bin/env python3
"""MCPAlign SFT warm-start training.

Trains the agent to output valid JSON actions and basic safety behaviors.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from mcpalign.models import get_lora_config
from mcpalign.utils import load_config, set_seed, setup_logging


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

    # Model & tokenizer — load manually to control device placement
    model_name = cfg["agent"]["model_name"]
    quant_cfg = cfg["agent"].get("quantization", {})

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg.get("load_in_4bit", True),
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg.get("bnb_4bit_compute_dtype", "float16")),
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
    )

    logger.info("Loading model: %s (4-bit)", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0},  # Force all on GPU 0, no CPU offload
        trust_remote_code=True,
    )

    # Prepare for kbit training (fixes gradient issues with quantized models)
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg.get("sft", {}).get("gradient_checkpointing", True)
    )

    # Apply LoRA
    lora_config = get_lora_config(cfg)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # SFT config
    sft_cfg = cfg.get("sft", {})
    training_args = SFTConfig(
        output_dir=sft_dir,
        num_train_epochs=sft_cfg.get("num_epochs", 3),
        per_device_train_batch_size=sft_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=sft_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=sft_cfg.get("learning_rate", 2e-5),
        max_length=sft_cfg.get("max_seq_length", sft_cfg.get("max_length", 2048)),
        save_steps=sft_cfg.get("save_steps", 200),
        logging_steps=sft_cfg.get("logging_steps", 10),
        bf16=True,
        max_grad_norm=sft_cfg.get("max_grad_norm", 1.0),
        warmup_ratio=sft_cfg.get("warmup_ratio", 0.1),
        lr_scheduler_type="cosine",
        gradient_checkpointing=sft_cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=sft_cfg.get("report_to", "none"),
        seed=seed,
    )

    # Train — pass model object, not string
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info("Model: %s", model_name)
    logger.info("SFT examples: %d", len(dataset))
    logger.info("Epochs: %d", sft_cfg.get("num_epochs", 3))
    logger.info("GPU memory: %d GB", torch.cuda.get_device_properties(0).total_memory // 1024**3)

    trainer.train()
    trainer.save_model(sft_dir)
    logger.info("SFT training complete. Checkpoint saved to %s", sft_dir)


if __name__ == "__main__":
    main()
