#!/usr/bin/env python3
"""MCPDefender GRPO Training Script.

Train the defender model using Group Relative Policy Optimization.
Uses TRL's GRPOTrainer with a deterministic reward function
(no LLM judge during training).

Usage:
    # fp16 (best quality, needs ~35GB VRAM):
    python scripts/mcptox_train_grpo.py --config configs/mcptox_defender.yaml \
        --no-sft-init --precision fp16

    # 4-bit (low VRAM, ~10GB):
    python scripts/mcptox_train_grpo.py --config configs/mcptox_defender.yaml \
        --no-sft-init --precision 4bit

    # From SFT checkpoint:
    python scripts/mcptox_train_grpo.py --config configs/mcptox_defender.yaml \
        --sft-checkpoint experiments/mcptox_defender/sft_checkpoint
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

from mcpalign.models import get_lora_config
from mcpalign.mcptox_reward import MCPDefenderReward
from mcpalign.utils import load_config, set_seed, setup_logging


def _parse_expected_tool(assistant_content: str) -> str:
    """Extract expected tool name from the SFT assistant response."""
    try:
        parsed = json.loads(assistant_content)
        return parsed.get("tool", "")
    except (json.JSONDecodeError, TypeError):
        import re
        m = re.search(r'"tool"\s*:\s*"([^"]+)"', assistant_content)
        return m.group(1) if m else ""


def build_grpo_dataset(sft_data: list) -> Dataset:
    """Build HF Dataset for GRPOTrainer from SFT data.

    Extracts prompt (system + user) and metadata for reward computation.
    The 'prompt' column is consumed by GRPOTrainer; extra columns are
    passed to the reward function via **kwargs.
    """
    records = []
    for example in sft_data:
        messages = example["messages"]
        if len(messages) < 3:
            continue

        # prompt = [system, user] (drop assistant)
        prompt = [messages[0], messages[1]]

        # Extract expected tool from the SFT target response
        expected_tool = _parse_expected_tool(messages[2]["content"])
        if not expected_tool:
            continue

        # Filter out extremely long prompts to prevent OOM
        prompt_len = sum(len(m.get("content", "")) for m in prompt)
        if prompt_len > 8000:
            continue

        records.append({
            "prompt": prompt,
            "expected_tool": expected_tool,
            "example_type": example.get("type", "benign"),
        })

    return Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser(description="MCPDefender GRPO Training")
    parser.add_argument(
        "--config", type=str, default="configs/mcptox_defender.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--base-model", type=str, default=None,
        help="Override base model (e.g. Qwen/Qwen2.5-3B-Instruct for quick test)",
    )
    parser.add_argument(
        "--sft-checkpoint", type=str, default=None,
        help="Path to SFT/DPO LoRA checkpoint to initialize from",
    )
    parser.add_argument(
        "--no-sft-init", action="store_true",
        help="Train from base model without SFT initialization",
    )
    parser.add_argument(
        "--precision", type=str, default="fp16",
        choices=["fp16", "8bit", "4bit"],
        help="Model precision: fp16, 8bit, or 4bit (default: fp16)",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed or cfg["experiment"].get("seed", 42)
    set_seed(seed)

    output_dir = cfg["experiment"]["output_dir"]
    grpo_dir = os.path.join(output_dir, "grpo_checkpoint")
    os.makedirs(grpo_dir, exist_ok=True)
    logger = setup_logging(output_dir, "train_grpo")

    logger.info("=" * 60)
    logger.info("MCPDefender — GRPO Training")
    logger.info("=" * 60)

    # ── Resolve model name ──────────────────────────────────────
    model_name = args.base_model or cfg["agent"]["model_name"]
    logger.info("Base model: %s", model_name)
    logger.info("Precision: %s", args.precision)

    # ── Load SFT data and build GRPO dataset ────────────────────
    sft_path = cfg["data"]["sft_data_path"]
    with open(sft_path) as f:
        sft_data = json.load(f)
    logger.info("Loaded %d SFT examples from %s", len(sft_data), sft_path)

    dataset = build_grpo_dataset(sft_data)
    logger.info("GRPO dataset: %d prompts", len(dataset))

    # ── Tokenizer ───────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ── Model loading ───────────────────────────────────────────
    # When passing model as string, GRPOTrainer loads it internally.
    # We pass model_init_kwargs to control precision.
    model_init_kwargs = {"trust_remote_code": True}

    if args.precision == "fp16":
        model_init_kwargs["torch_dtype"] = torch.float16
    elif args.precision == "8bit":
        model_init_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif args.precision == "4bit":
        quant_cfg = cfg["agent"].get("quantization", {})
        model_init_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(
                torch, quant_cfg.get("bnb_4bit_compute_dtype", "float16")
            ),
            bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
        )

    # SFT checkpoint handling
    sft_checkpoint = args.sft_checkpoint
    if not sft_checkpoint and not args.no_sft_init:
        default_sft = os.path.join(output_dir, "sft_checkpoint")
        if os.path.exists(default_sft):
            sft_checkpoint = default_sft
            logger.info("Found default SFT checkpoint: %s", sft_checkpoint)
        else:
            logger.warning("No SFT checkpoint found. Training from base model.")

    if sft_checkpoint and os.path.exists(sft_checkpoint):
        # Load base + SFT adapter, merge, pass model object
        logger.info("Loading SFT checkpoint: %s", sft_checkpoint)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map={"": 0}, **model_init_kwargs,
        )
        model = PeftModel.from_pretrained(base_model, sft_checkpoint)
        model = model.merge_and_unload()
        logger.info("SFT adapter merged.")
    else:
        # Pass model name as string — GRPOTrainer loads it
        model = model_name
        logger.info("Training from base model (no SFT init)")

    # ── Reward function ─────────────────────────────────────────
    reward_cfg = cfg.get("grpo", {}).get("reward", {})
    reward_fn = MCPDefenderReward(reward_cfg)

    # ── LoRA config ─────────────────────────────────────────────
    lora_config = get_lora_config(cfg)

    # ── GRPO config ─────────────────────────────────────────────
    grpo_cfg = cfg.get("grpo", {})

    use_vllm = grpo_cfg.get("use_vllm", False)

    grpo_kwargs = dict(
        output_dir=grpo_dir,
        num_train_epochs=grpo_cfg.get("num_train_epochs", 2),
        per_device_train_batch_size=grpo_cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=grpo_cfg.get("gradient_accumulation_steps", 4),
        num_generations=grpo_cfg.get("num_generations", 4),
        max_completion_length=grpo_cfg.get("max_completion_length", 256),
        learning_rate=grpo_cfg.get("learning_rate", 5e-6),
        lr_scheduler_type=grpo_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=grpo_cfg.get("warmup_ratio", 0.05),
        beta=grpo_cfg.get("beta", 0.0),
        max_grad_norm=grpo_cfg.get("max_grad_norm", 1.0),
        bf16=grpo_cfg.get("bf16", True),
        save_steps=grpo_cfg.get("save_steps", 100),
        logging_steps=grpo_cfg.get("logging_steps", 10),
        log_completions=grpo_cfg.get("log_completions", True),
        report_to=grpo_cfg.get("report_to", "none"),
        gradient_checkpointing=grpo_cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        model_init_kwargs=model_init_kwargs if isinstance(model, str) else None,
        torch_empty_cache_steps=1,
        seed=seed,
    )

    if use_vllm:
        grpo_kwargs["use_vllm"] = True
        grpo_kwargs["vllm_gpu_memory_utilization"] = grpo_cfg.get("vllm_gpu_memory_utilization", 0.3)

    # Disable Qwen3 think mode to save tokens (tool calls are short)
    grpo_kwargs["generation_kwargs"] = {
        "do_sample": True,
        "temperature": 0.7,
    }
    grpo_kwargs["chat_template_kwargs"] = {"enable_thinking": False}

    grpo_config = GRPOConfig(**grpo_kwargs)

    # ── Create trainer ──────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    logger.info("Training config:")
    logger.info("  Model: %s (%s)", model_name, args.precision)
    logger.info("  SFT init: %s", sft_checkpoint or "None")
    logger.info("  Dataset size: %d", len(dataset))
    logger.info("  Group size G=%d", grpo_cfg.get("num_generations", 4))
    logger.info("  Epochs: %d", grpo_cfg.get("num_train_epochs", 2))
    logger.info("  Batch size: %d", grpo_cfg.get("per_device_train_batch_size", 2))
    logger.info("  LR: %s", grpo_cfg.get("learning_rate", 5e-6))
    logger.info("  KL beta: %s", grpo_cfg.get("beta", 0.01))
    logger.info("  Max completion: %d", grpo_cfg.get("max_completion_length", 512))
    if torch.cuda.is_available():
        logger.info("  GPU: %s (%d GB free)",
                     torch.cuda.get_device_name(0),
                     torch.cuda.mem_get_info(0)[0] // 1024**3)

    # ── Train ───────────────────────────────────────────────────
    trainer.train()
    trainer.save_model(grpo_dir)
    logger.info("GRPO training complete. Checkpoint saved to %s", grpo_dir)


if __name__ == "__main__":
    main()
