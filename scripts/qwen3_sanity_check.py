#!/usr/bin/env python3
"""Qwen3-8B sanity check: verify chat template, Think tags, JSON parsing.

Pulls 10 MCPTox instances and runs Qwen3-8B on them in both Think and No-Think
modes. Checks:
  - Model loads with 4-bit quantization
  - Chat template applies cleanly
  - Output format: has <think>...</think> tags?
  - JSON extractable from response (with and without think block)
  - Compare to MCPTox's stored qwen3-8b_Think responses for same instances

Usage:
    python scripts/qwen3_sanity_check.py
"""

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from mcpalign.actions import parse_agent_action
from mcptox_defense_baseline import sample_instances, parse_tool_call

MCPTOX_PATH = "/work/tc442/MCPTox-Benchmark/response_all.json"
MODEL_NAME = "Qwen/Qwen3-8B"


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks from response."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def generate(model, tokenizer, system_prompt, query, enable_think=True, max_new_tokens=512):
    """Generate one response."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    # Qwen3 supports enable_thinking kwarg in apply_chat_template
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
    print("Loading MCPTox data ...")
    with open(MCPTOX_PATH) as f:
        data = json.load(f)

    # Get 10 mixed instances (e.g., 3 T1, 4 T2, 3 T3) from the seeded sample
    instances = sample_instances(data, max_per_cell=2, seed=42)
    sample = instances[:3] + instances[60:64] + instances[140:143]
    print(f"Selected {len(sample)} instances for sanity check")

    print(f"\nLoading {MODEL_NAME} (4-bit) ...")
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
    print("Model loaded.")

    # Also load MCPTox's pre-computed qwen3-8b_Think responses for comparison
    mcptox_responses = {}
    import re as _re
    for srv_name, srv in data["servers"].items():
        system_prompt = srv.get("clean_system_promot", "")
        prompt_order = _re.findall(r"Tool:\s*(\S+)\s*\n", system_prompt)
        cq = srv.get("clean_querys", [])
        q2t = {q.strip(): prompt_order[i] for i, q in enumerate(cq) if i < len(prompt_order)}
        for inst in srv.get("malicious_instance", []):
            for d in inst.get("datas", []):
                q = d.get("query", "").strip()
                t = q2t.get(q, "")
                if not t:
                    continue
                resp = d.get("online_result", {}).get("backend_model_responses", [{}])[0].get("qwen3-8b_Think")
                label = d.get("online_result", {}).get("labeled_model_results", [{}])[0].get("qwen3-8b_Think")
                if resp:
                    mcptox_responses[(srv_name, t, q)] = {"response": resp, "label": label}

    # Run both modes on each sample
    results = []
    for mode, enable_think in [("Think", True), ("No-Think", False)]:
        print(f"\n{'=' * 60}")
        print(f"Mode: {mode} (enable_thinking={enable_think})")
        print(f"{'=' * 60}")

        for i, inst in enumerate(sample):
            print(f"\n--- [{i}] {inst['paradigm']} / {inst['server']} / {inst['target_tool']} ---")
            print(f"  query: {inst['query'][:80]}")

            resp = generate(model, tokenizer, inst["system"], inst["query"],
                            enable_think=enable_think)

            # Check for think tags
            has_think = "<think>" in resp and "</think>" in resp
            stripped = strip_think(resp)

            # Parse tool call from both raw and stripped
            parsed_raw = parse_tool_call(resp)
            parsed_stripped = parse_tool_call(stripped)

            # Lookup MCPTox reference
            key = (inst["server"], inst["target_tool"], inst["query"])
            mcptox_ref = mcptox_responses.get(key, {})

            r = {
                "mode": mode,
                "idx": i,
                "paradigm": inst["paradigm"],
                "has_think": has_think,
                "response_len": len(resp),
                "stripped_len": len(stripped),
                "parsed_raw": parsed_raw,
                "parsed_stripped": parsed_stripped,
                "mcptox_ref_response": (mcptox_ref.get("response") or "")[:300],
                "mcptox_ref_label": mcptox_ref.get("label", ""),
                "our_response_preview": resp[:300],
                "our_stripped_preview": stripped[:300],
            }
            results.append(r)

            print(f"  has <think>: {has_think}")
            print(f"  response len: {len(resp)} (stripped: {len(stripped)})")
            print(f"  parsed tool (raw): {parsed_raw}")
            print(f"  parsed tool (stripped): {parsed_stripped}")
            print(f"  response preview: {resp[:200]}")
            if mcptox_ref:
                print(f"  MCPTox label: {mcptox_ref.get('label')}")

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")

    think = [r for r in results if r["mode"] == "Think"]
    nothink = [r for r in results if r["mode"] == "No-Think"]

    for label, rs in [("Think", think), ("No-Think", nothink)]:
        n = len(rs)
        has_think_count = sum(1 for r in rs if r["has_think"])
        parse_success_raw = sum(1 for r in rs if r["parsed_raw"])
        parse_success_stripped = sum(1 for r in rs if r["parsed_stripped"])
        avg_len = sum(r["response_len"] for r in rs) / max(n, 1)
        print(f"\n{label} mode:")
        print(f"  samples: {n}")
        print(f"  has <think> tags: {has_think_count}/{n}")
        print(f"  parse success (raw response): {parse_success_raw}/{n}")
        print(f"  parse success (stripped response): {parse_success_stripped}/{n}")
        print(f"  avg response length: {avg_len:.0f} chars")

    # Save
    out_dir = Path("experiments/qwen3_sanity")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_dir}/results.json")


if __name__ == "__main__":
    main()
