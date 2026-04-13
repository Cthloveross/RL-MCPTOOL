#!/usr/bin/env python3
"""Run Qwen3-8B bf16 (no quantization) on MCPTox full or sampled instances.

Purpose: check if 4-bit quantization is suppressing ASR.
MCPTox ref = 43.3%, our 4-bit = 22.0%. If bf16 → ~40%, quantization is the cause.

Usage:
    python scripts/eval_bf16_baseline.py --all          # full 1348 instances
    python scripts/eval_bf16_baseline.py                # 150 sample (default)
    python scripts/eval_bf16_baseline.py --skip-inference --tag bf16_150  # re-judge only
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
from transformers import AutoModelForCausalLM, AutoTokenizer

from mcpalign.llm_judge import LLMJudge
from mcptox_defense_baseline import sample_instances, apply_defense

MCPTOX_PATH = "/work/tc442/MCPTox-Benchmark/response_all.json"
OUTPUT_DIR = Path("experiments/mcptox_bf16")


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def load_model_bf16(model_name: str):
    print(f"Loading {model_name} (bf16, NO quantization)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded. GPU mem: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
    return model, tokenizer


def generate(model, tokenizer, system_prompt, query, enable_think=True, max_new_tokens=2048):
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
    ap.add_argument("--all", action="store_true", help="Use all 1348 instances instead of 150 sample")
    ap.add_argument("--modes", nargs="+", default=["no_defense", "prompt_hardening", "defensive_tokens"])
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--skip-inference", action="store_true")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or ("bf16_all" if args.all else "bf16_150")
    out_csv = OUTPUT_DIR / f"{tag}_raw.csv"

    # Load MCPTox
    print("Loading MCPTox...")
    with open(MCPTOX_PATH) as f:
        data = json.load(f)

    if args.all:
        # Use ALL instances (no sampling cap)
        instances = sample_instances(data, max_per_cell=9999, seed=args.seed)
    else:
        instances = sample_instances(data, max_per_cell=2, seed=args.seed)
    print(f"Using {len(instances)} instances")

    if not args.skip_inference:
        model, tokenizer = load_model_bf16(args.model)

        rows = []
        total = len(instances) * len(args.modes)
        step = 0
        for mode in args.modes:
            print(f"\n=== bf16 / {mode} ===")
            for i, inst in enumerate(instances):
                step += 1
                sys_prompt = apply_defense(inst["system"], mode)
                raw_resp = generate(model, tokenizer, sys_prompt, inst["query"],
                                    max_new_tokens=args.max_new_tokens)
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
                    "response_stripped": stripped,
                    "has_think": "<think>" in raw_resp,
                    "raw_len": len(raw_resp),
                    "stripped_len": len(stripped),
                })
                if step % 20 == 0 or step == total:
                    print(f"  [{step}/{total}] {mode} {i+1}/{len(instances)}")

        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
        print(f"\nSaved to {out_csv}")
        del model, tokenizer
        torch.cuda.empty_cache()
    else:
        df = pd.read_csv(out_csv)

    # LLM Judge
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
    df.to_csv(OUTPUT_DIR / f"{tag}_judged.csv", index=False)

    # Summary
    print("\n" + "=" * 60)
    print(f"bf16 — Per-Defense × Per-Paradigm ASR")
    print("=" * 60)
    pivot = df.pivot_table(
        values="llm_success", index="defense_mode",
        columns="paradigm", aggfunc="mean",
    ) * 100
    pivot["ALL"] = df.groupby("defense_mode")["llm_success"].mean() * 100
    print(pivot.round(1).to_string())

    print("\n" + "=" * 60)
    print("bf16 — Per-Defense × Per-Level ASR")
    print("=" * 60)
    pivot_l = df.pivot_table(
        values="llm_success", index="defense_mode",
        columns="level", aggfunc="mean",
    ) * 100
    print(pivot_l.round(1).to_string())

    # Compare to 4-bit
    print("\n" + "=" * 60)
    print("Comparison: bf16 vs 4-bit (no_defense)")
    print("=" * 60)
    if "no_defense" in pivot.index:
        bf16_row = pivot.loc["no_defense"]
        print(f"bf16:  T1={bf16_row.get('Template-1',0):.1f}  T2={bf16_row.get('Template-2',0):.1f}  T3={bf16_row.get('Template-3',0):.1f}  ALL={bf16_row.get('ALL',0):.1f}")
        print(f"4-bit: T1=22.2  T2=0.0  T3=40.3  ALL=22.0")
        print(f"MCPTox ref: T1=27.8  T2=18.3  T3=68.1  ALL=43.3")

    summary = {
        "tag": tag,
        "precision": "bf16",
        "n_instances": len(instances),
        "modes": args.modes,
        "per_paradigm_asr": pivot.round(2).to_dict(),
        "api_calls": api_calls,
        "cached_calls": cached_calls,
    }
    with open(OUTPUT_DIR / f"{tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nDone.")


if __name__ == "__main__":
    main()
