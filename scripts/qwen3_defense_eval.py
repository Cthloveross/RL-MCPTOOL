#!/usr/bin/env python3
"""Qwen3-8B defense baseline on MCPTox (saves full responses, strips think tags).

Differences from mcptox_defense_baseline.py:
  - Qwen3-8B (configurable via --model)
  - Saves FULL response (no 200-char truncation)
  - Automatically strips <think>...</think> blocks before parsing/judging
  - Supports --mode to run only a subset of defense conditions (e.g., skip no_defense
    when MCPTox labels will be used instead)
  - Integrates LLMJudge at the end (no separate rejudge step)

Usage:
    python scripts/qwen3_defense_eval.py                       # all 3 modes
    python scripts/qwen3_defense_eval.py --skip no_defense     # skip baseline
    python scripts/qwen3_defense_eval.py --model Qwen/Qwen3-8B
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from mcpalign.llm_judge import LLMJudge
from mcptox_defense_baseline import (
    sample_instances,
    apply_defense,
)


MCPTOX_PATH = "/work/tc442/MCPTox-Benchmark/response_all.json"
OUTPUT_DIR = Path("experiments/qwen3_defense_baseline")


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks."""
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def load_model(model_name: str):
    print(f"Loading {model_name} (4-bit) ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, system_prompt, query, enable_think=True, max_new_tokens=1024):
    """Generate with optional think mode (Qwen3 supports enable_thinking)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        kwargs["enable_thinking"] = enable_think
        prompt = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        prompt = tokenizer.apply_chat_template(messages, **kwargs)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--modes", nargs="+",
                    default=["no_defense", "prompt_hardening", "defensive_tokens"])
    ap.add_argument("--enable-think", action="store_true", default=True,
                    help="Enable Qwen3 think mode (default: True)")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-inference", action="store_true",
                    help="Skip inference, use existing CSV (for re-judging only)")
    args = ap.parse_args()

    enable_think = not args.no_think
    print(f"Think mode: {enable_think}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_short = args.model.split("/")[-1]
    out_csv = OUTPUT_DIR / f"{model_short}_{'think' if enable_think else 'nothink'}_raw.csv"

    # Load MCPTox data + regenerate the same 150 sample
    print("Loading MCPTox ...")
    with open(MCPTOX_PATH) as f:
        data = json.load(f)
    instances = sample_instances(data, max_per_cell=2, seed=args.seed)
    print(f"Sampled {len(instances)} instances")

    if not args.skip_inference:
        model, tokenizer = load_model(args.model)

        rows = []
        total = len(instances) * len(args.modes)
        step = 0
        for mode in args.modes:
            print(f"\n=== Defense mode: {mode} ===")
            for i, inst in enumerate(instances):
                step += 1
                sys_prompt = apply_defense(inst["system"], mode)
                raw_resp = generate(
                    model, tokenizer, sys_prompt, inst["query"],
                    enable_think=enable_think, max_new_tokens=args.max_new_tokens,
                )
                stripped = strip_think(raw_resp)
                rows.append({
                    "defense_mode": mode,
                    "server": inst["server"],
                    "paradigm": inst["paradigm"],
                    "level": inst["level"],
                    "target_tool": inst["target_tool"],
                    "poison_name": inst["poison_name"],
                    "query": inst["query"],
                    "poisoned_tool": inst.get("poisoned_tool", ""),
                    "security_risk_desc": inst.get("security_risk_desc", ""),
                    "response_raw": raw_resp,
                    "response_stripped": stripped,
                    "has_think": "<think>" in raw_resp,
                    "raw_len": len(raw_resp),
                    "stripped_len": len(stripped),
                })
                if step % 20 == 0 or step == total:
                    print(f"  [{step}/{total}] {mode} {i+1}/{len(instances)}")

        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
        print(f"\nSaved raw responses to {out_csv}")

        del model, tokenizer
        torch.cuda.empty_cache()
    else:
        print(f"Loading existing CSV: {out_csv}")
        df = pd.read_csv(out_csv)

    # LLM judge
    print(f"\n=== LLM Judge on {len(df)} rows ===")
    judge = LLMJudge()
    llm_labels, llm_success = [], []
    api_calls, cached_calls = 0, 0
    for i, row in df.iterrows():
        result = judge.judge(
            query=row["query"],
            poisoned_tool=row["poisoned_tool"],
            paradigm=row["paradigm"],
            model_response=row["response_stripped"] or "",
            legitimate_tool=row["target_tool"],
            malicious_tool=row["poison_name"],
            security_risk_desc=row.get("security_risk_desc", "") or "",
        )
        llm_labels.append(result.label)
        llm_success.append(result.label == "Success")
        if result.cached:
            cached_calls += 1
        else:
            api_calls += 1
        if (i + 1) % 50 == 0 or i == len(df) - 1:
            print(f"  [{i+1}/{len(df)}] api={api_calls} cached={cached_calls}")

    df["llm_label"] = llm_labels
    df["llm_success"] = llm_success
    df.to_csv(OUTPUT_DIR / f"{model_short}_{'think' if enable_think else 'nothink'}_judged.csv", index=False)

    # Summary
    print("\n" + "=" * 60)
    print("Per-Defense × Per-Paradigm ASR")
    print("=" * 60)
    pivot = df.pivot_table(
        values="llm_success", index="defense_mode",
        columns="paradigm", aggfunc="mean",
    ) * 100
    pivot["ALL"] = df.groupby("defense_mode")["llm_success"].mean() * 100
    print(pivot.round(1).to_string())

    print("\n" + "=" * 60)
    print("Per-Defense × Per-Level ASR")
    print("=" * 60)
    pivot_l = df.pivot_table(
        values="llm_success", index="defense_mode",
        columns="level", aggfunc="mean",
    ) * 100
    print(pivot_l.round(1).to_string())

    # Improvement vs no_defense
    if "no_defense" in pivot.index:
        print("\n" + "=" * 60)
        print("Defense improvement vs no_defense (pp reduction)")
        print("=" * 60)
        baseline = pivot.loc["no_defense"]
        for mode in pivot.index:
            if mode == "no_defense":
                continue
            red = baseline - pivot.loc[mode]
            print(f"{mode}: {red.round(1).to_dict()}")

    # Save summary
    summary = {
        "model": args.model,
        "enable_think": enable_think,
        "n_instances": len(instances),
        "modes": args.modes,
        "per_paradigm_asr": pivot.round(2).to_dict(),
        "per_level_asr": pivot_l.round(2).to_dict(),
        "think_tag_rate": float(df["has_think"].mean()) if "has_think" in df.columns else 0,
        "avg_raw_len": float(df["raw_len"].mean()) if "raw_len" in df.columns else 0,
        "avg_stripped_len": float(df["stripped_len"].mean()) if "stripped_len" in df.columns else 0,
    }
    with open(OUTPUT_DIR / f"{model_short}_{'think' if enable_think else 'nothink'}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nDone. Outputs in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
