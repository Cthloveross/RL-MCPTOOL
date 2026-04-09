"""Model loading utilities for attacker and victim models."""

import logging
import os
from typing import Optional

import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger("mcpoisoner")


def get_lora_config(cfg: dict) -> LoraConfig:
    """Build LoRA config from the attacker section of the experiment config."""
    lora_cfg = cfg["attacker"]["lora"]
    return LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_attacker_model(
    cfg: dict,
    checkpoint_path: Optional[str] = None,
    for_training: bool = False,
):
    """Load the attacker model (small LM) with optional LoRA checkpoint.

    Args:
        cfg: Full experiment config dict.
        checkpoint_path: Path to a saved LoRA adapter. If None, loads base model.
        for_training: If True, keep model trainable; if False, merge LoRA and eval().

    Returns:
        (model, tokenizer) tuple.
    """
    model_name = cfg["attacker"]["model_name"]
    dtype_str = cfg["attacker"].get("dtype", "bfloat16")
    dtype = getattr(torch, dtype_str)

    logger.info("Loading attacker model: %s (dtype=%s)", model_name, dtype_str)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info("Loading LoRA checkpoint from %s", checkpoint_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        if not for_training:
            model = model.merge_and_unload()
            model.eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True,
        )
        if not for_training:
            model.eval()

    return model, tokenizer


def load_victim_model(cfg: dict, model_key: str = "victim"):
    """Load a victim model with 4-bit quantization.

    Args:
        cfg: Full experiment config dict.
        model_key: Config key — "victim" or "transfer_victim".

    Returns:
        (model, tokenizer) tuple.
    """
    model_cfg = cfg[model_key]
    model_name = model_cfg["model_name"]
    quant_cfg = model_cfg.get("quantization", {})

    logger.info("Loading victim model: %s (4-bit quantized)", model_name)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg.get("load_in_4bit", True),
        bnb_4bit_compute_dtype=getattr(
            torch, quant_cfg.get("bnb_4bit_compute_dtype", "float16")
        ),
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    return model, tokenizer
