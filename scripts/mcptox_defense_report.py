#!/usr/bin/env python3
"""Quick report generator for MCPTox defense baseline results."""

import json
import sys
import pandas as pd
from pathlib import Path

OUT = Path("experiments/mcptox_defense_baseline")

def main():
    raw = pd.read_csv(OUT / "defense_baseline_raw.csv")
    print(f"Total rows: {len(raw)}")
    print(f"Modes: {sorted(raw['defense_mode'].unique())}")

    # Per-defense × per-paradigm
    print("\n=== ASR by defense × paradigm ===")
    pivot = raw.pivot_table(values="hijacked", index="defense_mode",
                            columns="paradigm", aggfunc="mean") * 100
    pivot["ALL"] = raw.groupby("defense_mode")["hijacked"].mean() * 100
    print(pivot.round(1).to_string())

    # Per-defense × per-level
    print("\n=== ASR by defense × level ===")
    pivot_l = raw.pivot_table(values="hijacked", index="defense_mode",
                              columns="level", aggfunc="mean") * 100
    print(pivot_l.round(1).to_string())

    # Tool vs arg tampering breakdown
    if "tool_hijacked" in raw.columns and "arg_tampered" in raw.columns:
        print("\n=== Tool hijack vs Arg tampering breakdown ===")
        for mode in sorted(raw["defense_mode"].unique()):
            sub = raw[raw["defense_mode"] == mode]
            tot = len(sub)
            hij = int(sub["hijacked"].sum())
            tool_hij = int(sub["tool_hijacked"].sum())
            arg_tamp = int(sub["arg_tampered"].sum())
            print(f"  {mode}: total_hij={hij}/{tot} ({hij/tot*100:.1f}%), tool_hij={tool_hij}, arg_tamp={arg_tamp}")

    # Parse error
    if "parse_success" in raw.columns:
        print(f"\nParse error rate: {(~raw['parse_success']).sum()}/{len(raw)} "
              f"({(~raw['parse_success']).mean()*100:.1f}%)")

    # Improvement over no_defense
    print("\n=== Improvement over no_defense (pp reduction) ===")
    if "no_defense" in pivot.index:
        baseline = pivot.loc["no_defense"]
        for mode in ["prompt_hardening", "defensive_tokens"]:
            if mode in pivot.index:
                red = baseline - pivot.loc[mode]
                print(f"{mode}: {red.round(1).to_dict()}")

    # Go/No-Go
    if "no_defense" in pivot.index and "Template-3" in pivot.columns:
        p3_base = pivot.loc["no_defense", "Template-3"]
        print(f"\n=== Go/No-Go ===")
        print(f"P3 no_defense ASR: {p3_base:.1f}%")
        if "prompt_hardening" in pivot.index:
            p3_ph = pivot.loc["prompt_hardening", "Template-3"]
            print(f"P3 prompt_hardening ASR: {p3_ph:.1f}%  (reduction: {p3_base - p3_ph:+.1f}pp)")
        if "defensive_tokens" in pivot.index:
            p3_dt = pivot.loc["defensive_tokens", "Template-3"]
            print(f"P3 defensive_tokens ASR: {p3_dt:.1f}%  (reduction: {p3_base - p3_dt:+.1f}pp)")


if __name__ == "__main__":
    main()
