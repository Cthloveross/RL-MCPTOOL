#!/usr/bin/env python3
"""Calibration reference: compute per-model ASR on our sampled 150 instances.

Uses MCPTox's pre-computed labels to show what ASR looks like for the 15
online models (DeepSeek-R1, Claude, GPT-4o-mini, etc.) on the EXACT same
subset that our defense baseline uses. This gives calibration targets for
our Qwen2.5-7B re-judged results.

Zero API calls, ~30s, pure lookup.

Usage:
    python scripts/mcptox_compare_to_mcptox.py
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))  # for sibling scripts

import pandas as pd

from mcptox_defense_baseline import sample_instances

MCPTOX_PATH = "/work/tc442/MCPTox-Benchmark/response_all.json"
OUTPUT_DIR = Path("experiments/mcptox_reference")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading MCPTox data ...")
    with open(MCPTOX_PATH) as f:
        data = json.load(f)

    print("Regenerating 150-sample subset (seed=42) ...")
    instances = sample_instances(data, max_per_cell=2, seed=42)
    print(f"  Got {len(instances)} instances")

    # Build lookup: for each instance, find its MCPTox entry and pull labels
    import re
    # Build reverse index: (server, paradigm, target_tool, query, poison_name) → labels dict
    index = {}
    for srv_name, srv in data["servers"].items():
        system_prompt = srv.get("clean_system_promot", "")
        prompt_order = re.findall(r"Tool:\s*(\S+)\s*\n", system_prompt)
        cq = srv.get("clean_querys", [])
        q2t = {q.strip(): prompt_order[i] for i, q in enumerate(cq) if i < len(prompt_order)}

        for inst in srv.get("malicious_instance", []):
            if inst.get("wrong_data") == 1:
                continue
            paradigm = inst.get("metadata", {}).get("paradigm", "")
            poison_match = re.search(r"Tool:\s*(\S+)", inst.get("poisoned_tool", ""))
            if not poison_match:
                continue
            poison_name = poison_match.group(1)

            for d in inst.get("datas", []):
                query = d.get("query", "").strip()
                target = q2t.get(query, "")
                if not target:
                    continue

                online = d.get("online_result", {})
                labels = {}
                if online.get("labeled_model_results"):
                    labels = online["labeled_model_results"][0]

                key = (srv_name, paradigm, target, poison_name, query)
                index[key] = labels

    # Lookup for each of our 150 sampled instances
    rows = []
    missing = 0
    for inst in instances:
        key = (inst["server"], inst["paradigm"], inst["target_tool"],
               inst["poison_name"], inst["query"])
        labels = index.get(key, {})
        if not labels:
            missing += 1
            continue
        for model, label in labels.items():
            rows.append({
                "server": inst["server"],
                "paradigm": inst["paradigm"],
                "level": inst["level"],
                "target_tool": inst["target_tool"],
                "model": model,
                "label": label,
                "is_success": label == "Success",
            })

    print(f"  Matched {len(instances) - missing}/{len(instances)} instances; {missing} missing labels")

    df = pd.DataFrame(rows)
    print(f"  Total (instance, model) rows: {len(df)}")

    # Per-model × per-paradigm ASR
    print("\n=== Per-Model × Per-Paradigm ASR (on our 150-sample subset) ===")
    pivot = df.pivot_table(
        values="is_success", index="model", columns="paradigm", aggfunc="mean",
    ) * 100
    pivot["ALL"] = df.groupby("model")["is_success"].mean() * 100
    pivot["n"] = df.groupby("model").size()
    pivot = pivot.sort_values("ALL", ascending=False)
    print(pivot.round(1).to_string())

    # Save
    pivot.to_csv(OUTPUT_DIR / "mcptox_reference_asr.csv")

    # Also per-model × per-level
    pivot_l = df.pivot_table(
        values="is_success", index="model", columns="level", aggfunc="mean",
    ) * 100
    pivot_l = pivot_l.reindex(pivot.index)
    pivot_l.to_csv(OUTPUT_DIR / "mcptox_reference_asr_by_level.csv")

    # Save summary
    summary = {
        "n_instances": len(instances),
        "n_mapped": len(instances) - missing,
        "paradigm_counts": df.groupby("paradigm").size().to_dict(),
        "models_count": len(pivot),
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nSaved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
