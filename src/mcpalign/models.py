"""Model loading utilities for MCPAlign agent training."""

import logging
import os
from typing import Optional

import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger("mcpalign")


def get_lora_config(cfg: dict) -> LoraConfig:
    """Build LoRA config from the agent section of the experiment config."""
    lora_cfg = cfg["agent"]["lora"]
    return LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_agent_model(cfg: dict, checkpoint_path: Optional[str] = None, for_training: bool = False):
    """Load the agent model with optional LoRA checkpoint.

    For MCPAlign, the 'agent' is the model being trained to resist attacks.
    It is loaded with 4-bit quantization (QLoRA) for memory efficiency.
    """
    model_name = cfg["agent"]["model_name"]
    quant_cfg = cfg["agent"].get("quantization", {})

    logger.info("Loading agent model: %s", model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb_config = None
    if quant_cfg.get("load_in_4bit", False):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(
                torch, quant_cfg.get("bnb_4bit_compute_dtype", "float16")
            ),
            bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
        )

    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info("Loading LoRA checkpoint from %s", checkpoint_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        if not for_training:
            model = model.merge_and_unload()
            model.eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        if not for_training:
            model.eval()

    return model, tokenizer
