#!/usr/bin/env python3
"""Judge the 450 Qwen3-8B responses produced by mcptox_run_qwen3.py.

Reads experiments/mcptox_qwen3/qwen3_responses_all.json, runs LLMJudge on
each (using the think-stripped `response` field), writes per-row labels and
prints per-defense x per-paradigm ASR table.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from mcpalign.llm_judge import LLMJudge

IN_JSON = Path("experiments/mcptox_qwen3/qwen3_responses_all.json")
OUT_DIR = Path("experiments/mcptox_qwen3")


def main():
    with open(IN_JSON) as f:
        rows = json.load(f)
    print(f"Loaded {len(rows)} rows")

    judge = LLMJudge()
    api, cached = 0, 0
    for i, r in enumerate(rows):
        result = judge.judge(
            query=r["query"],
            poisoned_tool=r["poisoned_tool"],
            paradigm=r["paradigm"],
            model_response=r.get("response") or "",
            legitimate_tool=r["target_tool"],
            malicious_tool=r["poison_name"],
            security_risk_desc=r.get("security_risk_desc", "") or "",
        )
        r["llm_label"] = result.label
        r["llm_success"] = result.label == "Success"
        if result.cached:
            cached += 1
        else:
            api += 1
        if (i + 1) % 50 == 0 or i == len(rows) - 1:
            print(f"  [{i+1}/{len(rows)}] api={api} cached={cached}")

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "qwen3_responses_all_judged.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")

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

    summary = {
        "n_rows": len(df),
        "api_calls": api,
        "cached_calls": cached,
        "per_paradigm_asr": pivot.round(2).to_dict(),
        "per_level_asr": pivot_l.round(2).to_dict(),
    }
    with open(OUT_DIR / "qwen3_responses_all_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved.")


if __name__ == "__main__":
    main()
