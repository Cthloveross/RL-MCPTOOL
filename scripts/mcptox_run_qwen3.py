#!/usr/bin/env python3
"""Run Qwen3-8B on MCPTox 150-sample subset under 3 defense conditions.

Differences from mcptox_defense_baseline.py:
  - Uses Qwen3-8B (not Qwen2.5-7B)
  - Saves FULL response (no 200-char truncation)
  - Strips <think>...</think> blocks before saving (MCPTox convention)
  - No in-script judging (use LLMJudge separately via mcptox_rejudge.py)

Usage:
    python scripts/mcptox_run_qwen3.py --mode no_defense
    python scripts/mcptox_run_qwen3.py --mode all  # run all 3 modes sequentially
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from mcpalign.utils import set_seed, setup_logging
from mcptox_defense_baseline import sample_instances, apply_defense

MCPTOX_PATH = "/work/tc442/MCPTox-Benchmark/response_all.json"
MODEL_NAME = "Qwen/Qwen3-8B"
OUTPUT_DIR = Path("experiments/mcptox_qwen3")

# Regex to strip think blocks
THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks (MCPTox stores responses without them)."""
    return THINK_RE.sub("", text).strip()


def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def run_inference(model, tokenizer, system_prompt, query,
                  enable_thinking=True, max_new_tokens=2048):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        kwargs["enable_thinking"] = enable_thinking
        prompt = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        prompt = tokenizer.apply_chat_template(messages, **kwargs)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=6144)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all",
                    choices=["all", "no_defense", "prompt_hardening", "defensive_tokens"])
    ap.add_argument("--enable-thinking", action="store_true", default=True,
                    help="Use Qwen3 Think mode (default). Use --no-think to disable.")
    ap.add_argument("--no-think", dest="enable_thinking", action="store_false")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    logger = setup_logging(str(OUTPUT_DIR), "qwen3_run")

    logger.info("=" * 60)
    logger.info("MCPTox Qwen3-8B inference")
    logger.info(f"Mode: {args.mode}, Think: {args.enable_thinking}")
    logger.info("=" * 60)

    # Load data + regenerate sample (seed 42)
    with open(MCPTOX_PATH) as f:
        data = json.load(f)
    instances = sample_instances(data, max_per_cell=2, seed=args.seed)
    logger.info(f"Sampled {len(instances)} instances")

    # Load model
    logger.info(f"Loading {MODEL_NAME} (4-bit) ...")
    model, tokenizer = load_model()
    logger.info("Model loaded")

    modes = ["no_defense", "prompt_hardening", "defensive_tokens"] \
            if args.mode == "all" else [args.mode]

    all_rows = []
    for mode in modes:
        logger.info("=" * 60)
        logger.info(f"Defense mode: {mode}")
        logger.info("=" * 60)

        for i, inst in enumerate(instances):
            system_prompt = apply_defense(inst["system"], mode)
            raw_response = run_inference(
                model, tokenizer, system_prompt, inst["query"],
                enable_thinking=args.enable_thinking,
                max_new_tokens=args.max_new_tokens,
            )
            stripped = strip_think(raw_response)

            row = {
                "defense_mode": mode,
                "server": inst["server"],
                "paradigm": inst["paradigm"],
                "level": inst["level"],
                "target_tool": inst["target_tool"],
                "poison_name": inst["poison_name"],
                "query": inst["query"],
                "poisoned_tool": inst.get("poisoned_tool", ""),
                "security_risk_desc": inst.get("security_risk_desc", ""),
                "raw_response": raw_response,
                "response": stripped,  # think-stripped, this is what we judge
                "raw_len": len(raw_response),
                "response_len": len(stripped),
                "had_think": "<think>" in raw_response,
            }
            all_rows.append(row)

            if (i + 1) % 20 == 0 or i == len(instances) - 1:
                think_count = sum(1 for r in all_rows if r["defense_mode"] == mode and r["had_think"])
                mode_n = sum(1 for r in all_rows if r["defense_mode"] == mode)
                logger.info(
                    "  [%s] %d/%d  think_tags=%d/%d  avg_raw_len=%d",
                    mode, i + 1, len(instances),
                    think_count, mode_n,
                    sum(r["raw_len"] for r in all_rows if r["defense_mode"] == mode) // max(mode_n, 1),
                )

    # Save
    out_json = OUTPUT_DIR / f"qwen3_responses_{'all' if args.mode == 'all' else args.mode}.json"
    with open(out_json, "w") as f:
        json.dump(all_rows, f, indent=2, default=str)
    logger.info(f"Saved {len(all_rows)} rows to {out_json}")

    # Also save a CSV without the full raw_response (for quick inspection)
    import pandas as pd
    df = pd.DataFrame(all_rows)
    csv_df = df.drop(columns=["raw_response"], errors="ignore").copy()
    csv_df["response_preview"] = csv_df["response"].str[:300]
    csv_df = csv_df.drop(columns=["response"], errors="ignore")
    csv_df.to_csv(OUTPUT_DIR / f"qwen3_responses_{'all' if args.mode == 'all' else args.mode}.csv", index=False)


if __name__ == "__main__":
    main()
