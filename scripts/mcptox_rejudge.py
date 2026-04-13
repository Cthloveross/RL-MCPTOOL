#!/usr/bin/env python3
"""Re-judge the existing 450-row defense baseline CSV with LLMJudge.

The original `mcptox_defense_baseline.py` produced
`experiments/mcptox_defense_baseline/defense_baseline_raw.csv` (150 instances
× 3 defense modes = 450 rows) using a broken keyword-based judge. This script:

1. Reproduces the exact 150-instance sample (seed=42)
2. Joins back to MCPTox to recover query / poisoned_tool / security_risk_desc
3. Asserts the CSV rows match the regenerated instances
4. Calls LLMJudge on every row (using response_preview as model output)
5. Audits truncation in response_preview
6. Writes corrected ASR tables

No new Qwen inference. ~450 API calls to gpt-4o-mini. ~$0.09. ~8 minutes.

Usage:
    python scripts/mcptox_rejudge.py
"""

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

from mcpalign.llm_judge import LLMJudge
from mcptox_defense_baseline import sample_instances

MCPTOX_PATH = "/work/tc442/MCPTox-Benchmark/response_all.json"
CSV_PATH = "experiments/mcptox_defense_baseline/defense_baseline_raw.csv"
OUTPUT_DIR = Path("experiments/mcptox_defense_baseline")


def looks_truncated(response_preview: str) -> bool:
    """Heuristic: does the response_preview end mid-JSON?"""
    s = (response_preview or "").strip()
    if len(s) < 10:
        return False
    # Full JSON must end with '}' or ']' or closing quote
    if s.endswith(("}", "]", '"', "```", "<|im_end|>")):
        return False
    # If it has an opening brace but no matching close
    open_braces = s.count("{")
    close_braces = s.count("}")
    if open_braces > close_braces:
        return True
    # Ends with comma, colon, quote — likely mid-field
    if s.endswith((",", ":", '"path"', '"arguments"', '"content"')):
        return True
    return False


def main():
    print("Loading MCPTox data ...")
    with open(MCPTOX_PATH) as f:
        data = json.load(f)

    print("Regenerating 150-sample subset (seed=42) ...")
    instances = sample_instances(data, max_per_cell=2, seed=42)
    print(f"  Got {len(instances)} instances")

    print("Loading CSV ...")
    df = pd.read_csv(CSV_PATH)
    print(f"  Loaded {len(df)} rows")

    # V1: Hard-assert the join is correct
    # CSV order: mode1×150, mode2×150, mode3×150
    modes = ["no_defense", "prompt_hardening", "defensive_tokens"]
    assert len(df) == 450, f"Expected 450 rows, got {len(df)}"
    for mode_idx, mode in enumerate(modes):
        mode_df = df[df["defense_mode"] == mode].reset_index(drop=True)
        assert len(mode_df) == 150, f"{mode} has {len(mode_df)} rows, expected 150"
        for i, inst in enumerate(instances):
            row = mode_df.iloc[i]
            assert row["server"] == inst["server"], \
                f"[{mode} row {i}] server mismatch: {row['server']} vs {inst['server']}"
            assert row["paradigm"] == inst["paradigm"], \
                f"[{mode} row {i}] paradigm mismatch: {row['paradigm']} vs {inst['paradigm']}"
            assert row["target_tool"] == inst["target_tool"], \
                f"[{mode} row {i}] target_tool mismatch"
            assert row["poison_name"] == inst["poison_name"], \
                f"[{mode} row {i}] poison_name mismatch"
    print("  V1 PASS: 450 rows join cleanly to 150 regenerated instances × 3 modes")

    # Join: add query, poisoned_tool, security_risk_desc to each row
    # by taking from the corresponding instance in order
    n_per_mode = 150
    df = df.copy()
    df["query"] = ""
    df["poisoned_tool"] = ""
    df["security_risk_desc"] = ""
    df["response_truncated"] = False

    for mode_idx, mode in enumerate(modes):
        start = mode_idx * n_per_mode
        for i, inst in enumerate(instances):
            idx = df[df["defense_mode"] == mode].index[i]
            df.at[idx, "query"] = inst["query"]
            df.at[idx, "poisoned_tool"] = inst.get("poisoned_tool", "")
            df.at[idx, "security_risk_desc"] = inst.get("security_risk_desc", "")
            df.at[idx, "response_truncated"] = looks_truncated(
                df.at[idx, "response_preview"]
            )

    # Truncation audit
    print("\n=== Truncation audit ===")
    for mode in modes:
        sub = df[df["defense_mode"] == mode]
        for p in ["Template-1", "Template-2", "Template-3"]:
            psub = sub[sub["paradigm"] == p]
            if len(psub) == 0:
                continue
            trunc = psub["response_truncated"].sum()
            pct = trunc / len(psub) * 100
            flag = " WARN" if pct > 20 else ""
            print(f"  {mode} / {p}: {trunc}/{len(psub)} truncated ({pct:.1f}%){flag}")

    # Run LLMJudge
    judge = LLMJudge()
    print(f"\nRunning LLMJudge on {len(df)} rows (cache: {len(judge._cache)} existing entries) ...")

    llm_labels = []
    llm_success = []
    api_calls = 0
    cached_calls = 0

    for i, row in df.iterrows():
        result = judge.judge(
            query=row["query"],
            poisoned_tool=row["poisoned_tool"],
            paradigm=row["paradigm"],
            model_response=row["response_preview"] or "",
            legitimate_tool=row["target_tool"],
            malicious_tool=row["poison_name"],
            security_risk_desc=row["security_risk_desc"],
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
    df["old_hijacked"] = df["hijacked"]

    # V5: T1 cross-check — old judge mostly worked on T1, so LLM should agree ≥ 80% on T1
    t1 = df[df["paradigm"] == "Template-1"]
    if len(t1) > 0:
        agree_t1 = (t1["llm_success"] == t1["old_hijacked"].astype(bool)).mean()
        print(f"\nV5 T1 cross-check: old vs new agreement on Template-1 = {agree_t1*100:.1f}%")

    # Compute ASR tables
    print("\n" + "=" * 60)
    print("CORRECTED RESULTS: Per-Defense × Per-Paradigm ASR (LLM judge)")
    print("=" * 60)
    pivot_new = df.pivot_table(
        values="llm_success", index="defense_mode",
        columns="paradigm", aggfunc="mean",
    ) * 100
    pivot_new["ALL"] = df.groupby("defense_mode")["llm_success"].mean() * 100
    print(pivot_new.round(1).to_string())

    print("\n" + "=" * 60)
    print("OLD (broken) RESULTS for comparison")
    print("=" * 60)
    pivot_old = df.pivot_table(
        values="old_hijacked", index="defense_mode",
        columns="paradigm", aggfunc="mean",
    ) * 100
    pivot_old["ALL"] = df.groupby("defense_mode")["old_hijacked"].mean() * 100
    print(pivot_old.round(1).to_string())

    print("\n" + "=" * 60)
    print("Per-Defense × Per-Level ASR (LLM judge)")
    print("=" * 60)
    pivot_level = df.pivot_table(
        values="llm_success", index="defense_mode",
        columns="level", aggfunc="mean",
    ) * 100
    print(pivot_level.round(1).to_string())

    print("\n" + "=" * 60)
    print("Defense improvement vs no_defense (pp reduction, LLM judge)")
    print("=" * 60)
    baseline = pivot_new.loc["no_defense"]
    for mode in ["prompt_hardening", "defensive_tokens"]:
        if mode in pivot_new.index:
            red = baseline - pivot_new.loc[mode]
            print(f"{mode}: {red.round(1).to_dict()}")

    # V7: Sanity check on baseline numbers
    print("\n" + "=" * 60)
    print("Sanity checks")
    print("=" * 60)
    nd = pivot_new.loc["no_defense"]
    t2_nd = nd.get("Template-2", 0)
    t3_nd = nd.get("Template-3", 0)
    print(f"  no_defense T2 ASR: {t2_nd:.1f}% (expect > 10% for Qwen2.5-7B)")
    print(f"  no_defense T3 ASR: {t3_nd:.1f}% (expect > 30% for Qwen2.5-7B)")
    if t2_nd < 10:
        print("  WARN: T2 ASR unexpectedly low. Investigate.")
    if t3_nd < 30:
        print("  WARN: T3 ASR unexpectedly low. Possible truncation issue.")

    # Adaptive train/test split decision
    print("\n" + "=" * 60)
    print("Train/test split decision")
    print("=" * 60)
    per_paradigm_baseline = {}
    for p in ["Template-1", "Template-2", "Template-3"]:
        per_paradigm_baseline[p] = float(nd.get(p, 0))
    print(f"Per-paradigm no_defense ASR: {per_paradigm_baseline}")
    hardest_paradigm = max(per_paradigm_baseline, key=per_paradigm_baseline.get)
    train_set = [p for p in ["Template-1", "Template-2", "Template-3"] if p != hardest_paradigm]
    print(f"Hardest paradigm (highest baseline ASR): {hardest_paradigm}")
    print(f"→ Held-out test paradigm: {hardest_paradigm}")
    print(f"→ Train paradigms: {train_set}")

    # Save everything
    df.to_csv(OUTPUT_DIR / "defense_baseline_rejudged.csv", index=False)
    pivot_new.to_csv(OUTPUT_DIR / "per_paradigm_asr_llm.csv")
    pivot_level.to_csv(OUTPUT_DIR / "per_level_asr_llm.csv")

    summary = {
        "n_rows": len(df),
        "api_calls": api_calls,
        "cached_calls": cached_calls,
        "per_paradigm_asr_new": pivot_new.round(2).to_dict(),
        "per_paradigm_asr_old_broken": pivot_old.round(2).to_dict(),
        "per_level_asr_new": pivot_level.round(2).to_dict(),
        "truncation_audit": {
            mode: {
                p: int(df[(df["defense_mode"] == mode) & (df["paradigm"] == p)]["response_truncated"].sum())
                for p in ["Template-1", "Template-2", "Template-3"]
            }
            for mode in modes
        },
        "train_test_split": {
            "hardest_paradigm": hardest_paradigm,
            "held_out_test": hardest_paradigm,
            "train_paradigms": train_set,
            "per_paradigm_baseline": per_paradigm_baseline,
        },
        "v5_t1_agreement": float(agree_t1) if len(t1) > 0 else None,
    }
    with open(OUTPUT_DIR / "rejudge_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nSaved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
