#!/usr/bin/env python3
"""
MCPTox Per-Tool-Type Vulnerability Re-Analysis (v2)
====================================================
Fixed: tool_names is alphabetically sorted but clean_querys follows
system prompt order. Parse tool order from system prompt instead.
"""

import json
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

MCPTOX_PATH = "/work/tc442/MCPTox-Benchmark/response_all.json"
OUTPUT_DIR = "experiments/mcptox_analysis"

# Models to exclude per MCPTox's own analysis.ipynb
EXCLUDE_MODELS = {"LLama-13b", "LLama-3-8b", "Qwen3-30b-A3b-No-Think",
                  "Deepseek-qwen3-8b-Think", "mistral"}


def extract_tool_order_from_prompt(system_prompt):
    """Extract tool names in order from the system prompt."""
    pattern = r"Tool:\s*(\S+)\s*\n"
    return re.findall(pattern, system_prompt)


def parse_mcptox(path):
    """Parse response_all.json with correct query→tool mapping."""
    with open(path) as f:
        data = json.load(f)

    rows = []
    tool_descriptions = {}

    for srv_name, srv in data["servers"].items():
        system_prompt = srv.get("clean_system_promot", "")
        clean_querys = srv.get("clean_querys", [])

        # Parse tool order from system prompt (NOT from tool_names which is alphabetical)
        prompt_tool_order = extract_tool_order_from_prompt(system_prompt)

        # Build query → tool mapping using prompt order
        query_to_tool = {}
        for i, q in enumerate(clean_querys):
            if i < len(prompt_tool_order):
                query_to_tool[q.strip()] = prompt_tool_order[i]

        # Extract descriptions
        for tn in prompt_tool_order:
            pat = rf"Tool:\s*{re.escape(tn)}\s*\nDescription:\s*(.*?)(?:\nArguments:|\nTool:|\Z)"
            m = re.search(pat, system_prompt, re.DOTALL)
            if m:
                tool_descriptions[tn] = m.group(1).strip()[:200]

        for inst in srv.get("malicious_instance", []):
            if inst.get("wrong_data") == 1:
                continue  # Skip bad data per MCPTox's own filtering

            metadata = inst.get("metadata", {})
            paradigm = metadata.get("paradigm", "unknown")
            security_risk = metadata.get("security risk", "unknown")

            for d in inst.get("datas", []):
                query = d.get("query", "").strip()
                target_tool = query_to_tool.get(query, "")

                # Merge local + online labels
                all_labels = {}
                for model, label in d.get("label", {}).items():
                    all_labels[model] = label
                for entry in d.get("online_result", {}).get("labeled_model_results", []):
                    if isinstance(entry, dict):
                        all_labels.update(entry)

                # Handle Qwen key deduplication (per MCPTox notebook)
                if "Qwen3-8b-Think" in all_labels and "qwen3-8b_Think" in all_labels:
                    del all_labels["Qwen3-8b-Think"]
                if "Qwen3-8b-No-Think" in all_labels and "qwen3-8b_NO_Think" in all_labels:
                    del all_labels["Qwen3-8b-No-Think"]
                if "Qwen3-14b-Think" in all_labels and "qwen3-14b_Think" in all_labels:
                    del all_labels["Qwen3-14b-Think"]
                if "Qwen3-14b-No-Think" in all_labels and "qwen3-14b_NO_Think" in all_labels:
                    del all_labels["Qwen3-14b-No-Think"]

                for model, label in all_labels.items():
                    if model in EXCLUDE_MODELS:
                        continue
                    rows.append({
                        "server": srv_name,
                        "target_tool": target_tool,
                        "paradigm": paradigm,
                        "security_risk": security_risk,
                        "model": model,
                        "label": label,
                        "is_success": label == "Success",
                    })

    return pd.DataFrame(rows), tool_descriptions


def annotate_tool_level(tool_name, description=""):
    """Heuristic annotation by permission level."""
    name = tool_name.lower()
    desc = (description or "").lower()
    combined = name + " " + desc

    # L5: Destructive
    if any(kw in combined for kw in [
        "delete", "remove", "drop", "revoke", "destroy", "purge", "uninstall",
        "wipe", "erase", "terminate", "close_issue", "close_pull",
    ]):
        return "L5"

    # L4: Execute/Privileged
    if any(kw in combined for kw in [
        "execut", "run_", "eval", "install", "deploy", "command",
        "sudo", "admin", "shell", "terminal", " sql",
        "create_table", "alter", "migrate", "sandbox", "run_code",
        "prisma_", "clickhouse",
    ]):
        return "L4"

    # L3: Communicate
    if any(kw in combined for kw in [
        "send", "email", "message", "post_", "share", "notify",
        "publish", "broadcast", "webhook", "slack", "chat",
        "comment", "reply", "tweet", "wechat", "draft",
        "reply_to", "create_post", "send_dm",
    ]):
        return "L3"

    # L2: Create/Modify
    if any(kw in combined for kw in [
        "write", "create", "add_", "update", "set_", "edit",
        "modify", "save", "upload", "put_", "patch", "insert",
        "append", "configure", "register", "store", "move",
        "rename", "copy", "fork", "merge", "push",
        "create_directory", "create_note", "create_branch",
        "create_issue", "create_pull", "create_repository",
        "open_tab", "navigate", "generate",
    ]):
        return "L2"

    # L1: Read-Only
    if any(kw in combined for kw in [
        "read", "get", "list", "search", "fetch", "query",
        "find", "show", "view", "check", "look", "browse",
        "download", "retrieve", "info", "stat", "describe",
        "count", "measure", "analyze", "convert", "calculat",
        "translat", "summariz", "extract", "parse", "resolve",
        "scrape", "crawl", "screenshot", "click", "scroll",
        "select", "evaluat", "predict", "classify", "detect",
        "geocod", "direction", "distance", "price", "rate",
        "quote", "profile", "balance", "histor", "overview",
        "income", "earning", "dividend", "insider", "financ",
        "market", "stock", "etf", "crypto", "currency",
        "address", "place", "weather", "route", "transit",
        "keyword", "paper", "arxiv", "brave_",
        "sequentialthinking", "recall", "memory",
    ]):
        return "L1"

    return "unclear"


def analyze(df, annotations, tool_descriptions, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    df["level"] = df["target_tool"].map(
        lambda t: annotations.get(t, {}).get("level", "unclear")
    )
    df_all = df[df["target_tool"] != ""].copy()
    df_labeled = df_all[df_all["level"] != "unclear"].copy()

    # Coverage stats
    total_tools = df_all["target_tool"].nunique()
    mapped_tools = df_labeled["target_tool"].nunique()
    unclear_tools = df_all[df_all["level"] == "unclear"]["target_tool"].nunique()

    print(f"Total rows: {len(df)}")
    print(f"Rows with target_tool mapped: {len(df_all)} ({len(df_all)/len(df)*100:.1f}%)")
    print(f"Rows with level annotation: {len(df_labeled)} ({len(df_labeled)/len(df)*100:.1f}%)")
    print(f"Tools: {total_tools} mapped, {unclear_tools} unclear")

    # Level distribution
    level_dist = df_labeled.groupby("level")["target_tool"].nunique()
    print(f"\n=== Tool count per level ===")
    for lvl in ["L1", "L2", "L3", "L4", "L5"]:
        print(f"  {lvl}: {level_dist.get(lvl, 0)} tools")

    # ── Per-level ASR ───────────────────────────────────────
    level_asr = df_labeled.groupby("level")["is_success"].agg(["mean", "sum", "count"])
    level_asr.columns = ["asr", "successes", "total"]
    print(f"\n=== Per-Level ASR (all models) ===")
    for lvl in ["L1", "L2", "L3", "L4", "L5"]:
        if lvl in level_asr.index:
            r = level_asr.loc[lvl]
            print(f"  {lvl}: {r['asr']*100:.1f}% ({int(r['successes'])}/{int(r['total'])})")

    # ── Per-level × Per-model ───────────────────────────────
    pivot = df_labeled.pivot_table(
        values="is_success", index="level", columns="model", aggfunc="mean"
    )
    pivot.to_csv(os.path.join(output_dir, "level_model_asr.csv"))
    print(f"\n=== Per-Model L1 vs L4 Gap ===")
    model_gaps = {}
    for model in sorted(pivot.columns):
        l1 = pivot.loc["L1", model] if "L1" in pivot.index else None
        l4 = pivot.loc["L4", model] if "L4" in pivot.index else None
        if l1 is not None and l4 is not None and not (np.isnan(l1) or np.isnan(l4)):
            gap = (l1 - l4) * 100
            model_gaps[model] = gap
            print(f"  {model:40s}: L1={l1*100:.1f}%, L4={l4*100:.1f}%, Gap={gap:+.1f}pp")

    # ── Spearman ────────────────────────────────────────────
    level_map = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5}
    per_tool = df_labeled.groupby(["target_tool", "level"]).agg(
        asr=("is_success", "mean"),
    ).reset_index()
    per_tool["level_num"] = per_tool["level"].map(level_map)
    rho, pval = stats.spearmanr(per_tool["level_num"], per_tool["asr"])
    print(f"\n=== Spearman Correlation ===")
    print(f"  rho = {rho:.3f}, p = {pval:.6f}")

    # ── Chi-square L1 vs L4 ────────────────────────────────
    l1l4 = df_labeled[df_labeled["level"].isin(["L1", "L4"])]
    p_chi = 1.0
    chi2 = 0
    if len(l1l4) > 0:
        contingency = pd.crosstab(l1l4["level"], l1l4["is_success"])
        chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)
        print(f"\n=== Chi-Square L1 vs L4 ===")
        print(f"  chi2 = {chi2:.2f}, p = {p_chi:.8f}")

    # ── Paradigm × Level ───────────────────────────────────
    paradigm_level = df_labeled.pivot_table(
        values="is_success", index="level", columns="paradigm", aggfunc="mean"
    )
    paradigm_level.to_csv(os.path.join(output_dir, "paradigm_level_asr.csv"))
    print(f"\n=== Paradigm × Level ASR ===")
    print((paradigm_level * 100).round(1).to_string())

    # ── Top/bottom tools ────────────────────────────────────
    tool_asr = df_labeled.groupby(["target_tool", "level"])["is_success"].agg(
        ["mean", "count"]
    ).reset_index()
    tool_asr.columns = ["tool", "level", "asr", "n"]
    tool_asr = tool_asr.sort_values("asr", ascending=False)
    tool_asr.to_csv(os.path.join(output_dir, "per_tool_asr.csv"), index=False)

    print(f"\n=== Top 10 Most Vulnerable Tools ===")
    for _, row in tool_asr.head(10).iterrows():
        print(f"  {row['tool']:35s} [{row['level']}]: {row['asr']*100:5.1f}% (n={int(row['n'])})")

    print(f"\n=== Bottom 10 Least Vulnerable Tools ===")
    for _, row in tool_asr.tail(10).iterrows():
        print(f"  {row['tool']:35s} [{row['level']}]: {row['asr']*100:5.1f}% (n={int(row['n'])})")

    # ══════════════════════════════════════════════════════════
    # Success Criteria
    # ══════════════════════════════════════════════════════════
    l1_asr = level_asr.loc["L1", "asr"] * 100 if "L1" in level_asr.index else 0
    l4_asr = level_asr.loc["L4", "asr"] * 100 if "L4" in level_asr.index else 0
    gap = l1_asr - l4_asr
    models_with_gap = sum(1 for g in model_gaps.values() if g > 10)

    print(f"\n{'='*60}")
    print("GO/NO-GO DECISION")
    print(f"{'='*60}")
    print(f"  [{'PASS' if gap > 10 else 'FAIL'}] L1-L4 gap > 10pp: {gap:+.1f}pp")
    print(f"  [{'PASS' if rho < -0.3 and pval < 0.05 else 'FAIL'}] Spearman < -0.3: rho={rho:.3f}, p={pval:.4f}")
    print(f"  [{'PASS' if models_with_gap >= 3 else 'FAIL'}] ≥3 models with gap>10pp: {models_with_gap}")
    print(f"  [{'PASS' if p_chi < 0.01 else 'FAIL'}] Chi-square p<0.01: p={p_chi:.6f}")

    # Save summary
    summary = {
        "per_level_asr": {lvl: float(level_asr.loc[lvl, "asr"]) for lvl in level_asr.index},
        "l1_asr": float(l1_asr), "l4_asr": float(l4_asr), "gap": float(gap),
        "spearman_rho": float(rho), "spearman_p": float(pval),
        "chi2": float(chi2), "chi2_p": float(p_chi),
        "models_with_gap_gt_10": int(models_with_gap),
        "total_rows": int(len(df_labeled)),
        "unique_tools": int(mapped_tools),
        "unclear_tools": int(unclear_tools),
        "coverage": float(len(df_all) / max(len(df), 1)),
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ══════════════════════════════════════════════════════════
    # Figures
    # ══════════════════════════════════════════════════════════
    levels = ["L1", "L2", "L3", "L4", "L5"]
    colors = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71", "#9b59b6"]
    asrs = [level_asr.loc[l, "asr"] * 100 if l in level_asr.index else 0 for l in levels]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(levels, asrs, color=colors)
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_xlabel("Permission Level")
    ax.set_title("MCPTox ASR by Tool Permission Level (Fixed Mapping)")
    for bar, asr in zip(bars, asrs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{asr:.1f}%", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "level_asr_bar.pdf"), dpi=150)
    fig.savefig(os.path.join(output_dir, "level_asr_bar.png"), dpi=150)
    plt.close()

    # Unclear tools list for manual review
    unclear = [t for t, v in annotations.items() if v["level"] == "unclear"]
    if unclear:
        print(f"\n=== Unclear tools ({len(unclear)}) — need manual review ===")
        for t in sorted(unclear):
            desc = tool_descriptions.get(t, "")[:80]
            print(f"  {t:35s}: {desc}")

    # Save annotations
    ann_df = pd.DataFrame([
        {"tool": t, "level": v["level"], "description": v.get("description", "")[:100]}
        for t, v in sorted(annotations.items())
    ])
    ann_df.to_csv(os.path.join(output_dir, "tool_annotations.csv"), index=False)

    print(f"\nResults saved to {output_dir}/")


def main():
    print("=" * 60)
    print("MCPTox Re-Analysis v2 (Fixed Query→Tool Mapping)")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n>>> Parsing MCPTox data (using system prompt tool order)...")
    df, tool_descriptions = parse_mcptox(MCPTOX_PATH)
    mapped = df[df["target_tool"] != ""]
    print(f"Parsed {len(df)} rows, {len(mapped)} mapped ({len(mapped)/len(df)*100:.1f}%)")
    print(f"Unique tools: {mapped['target_tool'].nunique()}, Models: {df['model'].nunique()}")

    print("\n>>> Annotating tools...")
    annotations = {}
    for tool in df["target_tool"].unique():
        if tool:
            desc = tool_descriptions.get(tool, "")
            annotations[tool] = {"level": annotate_tool_level(tool, desc), "description": desc}

    print("\n>>> Analysis...")
    analyze(df, annotations, tool_descriptions, OUTPUT_DIR)


if __name__ == "__main__":
    main()
