#!/usr/bin/env python3
"""MCPAlign DPO baseline training.

Trains the agent with offline DPO using preference pairs.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from datasets import Dataset
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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
        "chosen": [{"role": "assistant", "content": d["chosen"]}],
        "rejected": [{"role": "assistant", "content": d["rejected"]}],
    } for d in dpo_data])

    # Load model manually
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

    logger.info("Loading base model: %s (4-bit)", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )

    # Load SFT LoRA adapter if exists
    if os.path.exists(sft_checkpoint):
        logger.info("Loading SFT adapter from: %s", sft_checkpoint)
        model = PeftModel.from_pretrained(model, sft_checkpoint, is_trainable=True)
    else:
        logger.info("No SFT checkpoint found, starting from base model")
        dpo_cfg_section = cfg.get("dpo", {})
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=dpo_cfg_section.get("gradient_checkpointing", True)
        )
        lora_config = get_lora_config(cfg)
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # DPO config
    dpo_cfg = cfg.get("dpo", {})
    training_args = DPOConfig(
        output_dir=dpo_dir,
        num_train_epochs=dpo_cfg.get("num_epochs", 1),
        per_device_train_batch_size=dpo_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=dpo_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=dpo_cfg.get("learning_rate", 5e-7),
        beta=dpo_cfg.get("beta", 0.1),
        max_length=dpo_cfg.get("max_length", 2048),
        save_steps=dpo_cfg.get("save_steps", 200),
        logging_steps=dpo_cfg.get("logging_steps", 10),
        bf16=True,
        max_grad_norm=dpo_cfg.get("max_grad_norm", 0.3),
        warmup_ratio=dpo_cfg.get("warmup_ratio", 0.1),
        lr_scheduler_type="cosine",
        gradient_checkpointing=dpo_cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=dpo_cfg.get("report_to", "none"),
        seed=seed,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    logger.info("DPO Config: beta=%.2f, lr=%s, epochs=%d, pairs=%d",
                dpo_cfg.get("beta", 0.1), dpo_cfg.get("learning_rate", 5e-7),
                dpo_cfg.get("num_epochs", 1), len(dataset))
    logger.info("GPU memory: %d GB", torch.cuda.get_device_properties(0).total_memory // 1024**3)

    trainer.train()
    trainer.save_model(dpo_dir)
    logger.info("DPO training complete. Checkpoint saved to %s", dpo_dir)


if __name__ == "__main__":
    main()
