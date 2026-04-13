#!/usr/bin/env python3
"""Evaluate SFT/DPO trained models on MCPTox 150-instance test set.

Loads base Qwen3-8B + LoRA adapter, runs Think-mode inference on
the same 150 instances used for baseline, then LLM-judges the results.

Usage:
    python scripts/eval_trained_model.py --adapter experiments/mcptox_defender/sft_checkpoint --tag sft
    python scripts/eval_trained_model.py --adapter experiments/mcptox_defender/dpo_checkpoint --tag dpo
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
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from mcpalign.llm_judge import LLMJudge
from mcptox_defense_baseline import sample_instances

MCPTOX_PATH = "/work/tc442/MCPTox-Benchmark/response_all.json"
OUTPUT_DIR = Path("experiments/mcptox_defender")


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def load_model(base_model: str, adapter_path: str | None):
    print(f"Loading {base_model} (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb_config,
        device_map={"": 0}, trust_remote_code=True,
    )

    if adapter_path and os.path.exists(adapter_path):
        print(f"Loading LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print("Adapter merged.")

    model.eval()
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
    ap.add_argument("--base-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--adapter", required=True, help="Path to LoRA adapter checkpoint")
    ap.add_argument("--tag", required=True, help="Tag for output files (e.g. sft, dpo)")
    ap.add_argument("--modes", nargs="+", default=["no_defense"])
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-inference", action="store_true")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUT_DIR / f"{args.tag}_raw.csv"

    # Load MCPTox + same 150 sample
    print("Loading MCPTox...")
    with open(MCPTOX_PATH) as f:
        data = json.load(f)
    instances = sample_instances(data, max_per_cell=2, seed=args.seed)
    print(f"Sampled {len(instances)} instances")

    if not args.skip_inference:
        model, tokenizer = load_model(args.base_model, args.adapter)

        rows = []
        total = len(instances) * len(args.modes)
        step = 0
        for mode in args.modes:
            print(f"\n=== {args.tag} / {mode} ===")
            for i, inst in enumerate(instances):
                step += 1
                # Apply defense to system prompt
                if mode == "no_defense":
                    sys_prompt = inst["system"]
                elif mode == "prompt_hardening":
                    from mcptox_defense_baseline import apply_defense
                    sys_prompt = apply_defense(inst["system"], mode)
                elif mode == "defensive_tokens":
                    from mcptox_defense_baseline import apply_defense
                    sys_prompt = apply_defense(inst["system"], mode)
                else:
                    sys_prompt = inst["system"]

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
        print(f"Loading existing CSV: {out_csv}")
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

    judged_csv = OUTPUT_DIR / f"{args.tag}_judged.csv"
    df.to_csv(judged_csv, index=False)

    # Summary
    print("\n" + "=" * 60)
    print(f"{args.tag.upper()} — Per-Defense × Per-Paradigm ASR")
    print("=" * 60)
    pivot = df.pivot_table(
        values="llm_success", index="defense_mode",
        columns="paradigm", aggfunc="mean",
    ) * 100
    pivot["ALL"] = df.groupby("defense_mode")["llm_success"].mean() * 100
    print(pivot.round(1).to_string())

    print("\n" + "=" * 60)
    print(f"{args.tag.upper()} — Per-Defense × Per-Level ASR")
    print("=" * 60)
    pivot_l = df.pivot_table(
        values="llm_success", index="defense_mode",
        columns="level", aggfunc="mean",
    ) * 100
    print(pivot_l.round(1).to_string())

    # Save summary
    summary = {
        "tag": args.tag,
        "adapter": args.adapter,
        "n_instances": len(instances),
        "modes": args.modes,
        "per_paradigm_asr": pivot.round(2).to_dict(),
        "per_level_asr": pivot_l.round(2).to_dict(),
        "api_calls": api_calls,
        "cached_calls": cached_calls,
    }
    with open(OUTPUT_DIR / f"{args.tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nDone. Results in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
