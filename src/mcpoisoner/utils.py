"""Utility functions: config loading, seeding, logging setup."""

import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logging(output_dir: str, name: str = "mcpoisoner") -> logging.Logger:
    """Configure logging to both file and console."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"{name}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def get_device() -> torch.device:
    """Return the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def gpu_memory_summary() -> str:
    """Return a human-readable GPU memory summary."""
    if not torch.cuda.is_available():
        return "No GPU available"
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_mem / 1e9
    return (
        f"GPU: {torch.cuda.get_device_name(0)} | "
        f"Allocated: {allocated:.1f}GB | Reserved: {reserved:.1f}GB | Total: {total:.1f}GB"
    )
