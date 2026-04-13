#!/usr/bin/env python3
"""Validate LLMJudge against MCPTox pre-computed ground-truth labels.

Pipeline:
  1. Load MCPTox response_all.json
  2. Build validation pool: (paradigm, query, poisoned_tool, mcptox_model, response, gold)
     excluding gpt-4o-mini responses (judge is gpt-4o-mini) to avoid circularity
  3. Stratified sample: 50 per paradigm = 150 total
  4. Run LLMJudge on each, compare to gold
  5. Report overall agreement, per-paradigm, 4x4 confusion matrix, disagreements

Usage:
    python scripts/mcptox_validate_judge.py
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from mcpalign.llm_judge import LLMJudge, LABEL_SET


MCPTOX_PATH = "/work/tc442/MCPTox-Benchmark/response_all.json"
OUTPUT_DIR = Path("experiments/judge_validation")

# Exclude judge's own model to avoid self-validation circularity
EXCLUDE_MODELS = {"gpt-4o-mini"}

# Strong models prioritized for validation (most reliable as ground-truth reference)
PREFERRED_MODELS = [
    "claude-3-7-sonnet",
    "DeepSeek-R1",
    "qwen3-32b_Think",
    "Llama-3.1-70B-Instruct",
    "DeepSeek-v3",
    "o1-mini",
    "gemini-2.5-flash",
]


def build_validation_pool(data, seed=42):
    """Build flat list of (paradigm, ..., mcptox_model, response, gold) rows."""
    import re
    rng = random.Random(seed)
    pool = []

    for srv_name, srv in data["servers"].items():
        system_prompt = srv.get("clean_system_promot", "")
        clean_querys = srv.get("clean_querys", [])
        # Parse tool order from system prompt
        prompt_order = re.findall(r"Tool:\s*(\S+)\s*\n", system_prompt)
        q2t = {q.strip(): prompt_order[i] for i, q in enumerate(clean_querys) if i < len(prompt_order)}

        for inst in srv.get("malicious_instance", []):
            if inst.get("wrong_data") == 1:
                continue
            paradigm = inst.get("metadata", {}).get("paradigm", "")
            if paradigm not in ("Template-1", "Template-2", "Template-3"):
                continue

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
                responses = {}
                if online.get("backend_model_responses"):
                    responses = online["backend_model_responses"][0]

                for model_name in labels:
                    if model_name in EXCLUDE_MODELS:
                        continue
                    gold = labels.get(model_name)
                    if gold not in LABEL_SET:
                        continue
                    resp = responses.get(model_name, "")
                    if not isinstance(resp, str) or not resp.strip():
                        continue

                    pool.append({
                        "server": srv_name,
                        "paradigm": paradigm,
                        "query": query,
                        "target_tool": target,
                        "poison_name": poison_name,
                        "poisoned_tool": inst.get("poisoned_tool", ""),
                        "security_risk_desc": inst.get("security_risk_description", ""),
                        "mcptox_model": model_name,
                        "response": resp,
                        "gold": gold,
                    })

    rng.shuffle(pool)
    return pool


def stratified_sample(pool, n_per_paradigm=50, seed=42):
    """Sample n per paradigm, preferring strong models for reliability."""
    rng = random.Random(seed)
    by_paradigm = defaultdict(list)
    for row in pool:
        by_paradigm[row["paradigm"]].append(row)

    # Within each paradigm: prefer strong models first, then balance
    sampled = []
    for p in ["Template-1", "Template-2", "Template-3"]:
        candidates = by_paradigm.get(p, [])
        if not candidates:
            continue
        # Split by strong vs other, prefer strong
        strong = [r for r in candidates if r["mcptox_model"] in PREFERRED_MODELS]
        other = [r for r in candidates if r["mcptox_model"] not in PREFERRED_MODELS]
        rng.shuffle(strong)
        rng.shuffle(other)
        # Also balance across gold labels (avoid all-Success or all-Fail)
        by_gold = defaultdict(list)
        for r in (strong + other):
            by_gold[r["gold"]].append(r)
        # Round-robin across gold labels
        taken = []
        gold_keys = sorted(by_gold.keys())
        idx = {k: 0 for k in gold_keys}
        while len(taken) < n_per_paradigm and any(idx[k] < len(by_gold[k]) for k in gold_keys):
            for k in gold_keys:
                if idx[k] < len(by_gold[k]):
                    taken.append(by_gold[k][idx[k]])
                    idx[k] += 1
                    if len(taken) >= n_per_paradigm:
                        break
        sampled.extend(taken[:n_per_paradigm])
    return sampled


def run_validation(sample, judge):
    results = []
    n = len(sample)
    for i, row in enumerate(sample, 1):
        r = judge.judge(
            query=row["query"],
            poisoned_tool=row["poisoned_tool"],
            paradigm=row["paradigm"],
            model_response=row["response"],
            legitimate_tool=row["target_tool"],
            malicious_tool=row["poison_name"],
            security_risk_desc=row["security_risk_desc"],
        )
        results.append({
            **row,
            "llm_label": r.label,
            "agree": r.label == row["gold"],
            "cached": r.cached,
        })
        if i % 20 == 0 or i == n:
            ok = sum(1 for x in results if x["agree"])
            print(f"  [{i}/{n}] agreement so far: {ok/i*100:.1f}%")
    return pd.DataFrame(results)


def summarize(df):
    summary = {}
    summary["n"] = len(df)
    summary["overall_agreement"] = float(df["agree"].mean())

    per_paradigm = {}
    for p in sorted(df["paradigm"].unique()):
        sub = df[df["paradigm"] == p]
        per_paradigm[p] = {
            "n": int(len(sub)),
            "agreement": float(sub["agree"].mean()),
        }
    summary["per_paradigm"] = per_paradigm

    # Confusion matrix (gold rows × llm cols)
    labels = sorted(LABEL_SET)
    conf = pd.crosstab(df["gold"], df["llm_label"], dropna=False).reindex(
        index=labels, columns=labels, fill_value=0,
    )
    summary["confusion_matrix"] = conf.to_dict()

    # Per-paradigm confusion for quick eye-balling
    summary["per_paradigm_conf"] = {}
    for p in sorted(df["paradigm"].unique()):
        sub = df[df["paradigm"] == p]
        pc = pd.crosstab(sub["gold"], sub["llm_label"], dropna=False).reindex(
            index=labels, columns=labels, fill_value=0,
        )
        summary["per_paradigm_conf"][p] = pc.to_dict()

    return summary, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-paradigm", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading MCPTox data ...")
    with open(MCPTOX_PATH) as f:
        data = json.load(f)

    print("Building validation pool ...")
    pool = build_validation_pool(data, seed=args.seed)
    print(f"  Pool size: {len(pool)}")
    from collections import Counter
    print(f"  By paradigm: {Counter(r['paradigm'] for r in pool)}")
    print(f"  By gold: {Counter(r['gold'] for r in pool)}")

    sample = stratified_sample(pool, n_per_paradigm=args.n_per_paradigm, seed=args.seed)
    print(f"  Sampled: {len(sample)}")
    print(f"  Sample paradigms: {Counter(r['paradigm'] for r in sample)}")
    print(f"  Sample gold: {Counter(r['gold'] for r in sample)}")

    judge = LLMJudge()

    print("\nRunning LLMJudge on validation sample ...")
    df = run_validation(sample, judge)

    summary, conf = summarize(df)

    # Print
    print("\n" + "=" * 60)
    print("JUDGE VALIDATION RESULTS")
    print("=" * 60)
    print(f"Overall agreement: {summary['overall_agreement']*100:.1f}% (n={summary['n']})")
    print(f"\nPer-paradigm agreement:")
    for p, v in summary["per_paradigm"].items():
        print(f"  {p}: {v['agreement']*100:.1f}% (n={v['n']})")

    print(f"\nConfusion matrix (rows=gold, cols=llm):")
    print(conf.to_string())

    # Save
    df.to_csv(OUTPUT_DIR / "validation_raw.csv", index=False)
    with open(OUTPUT_DIR / "agreement.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    conf.to_csv(OUTPUT_DIR / "confusion_matrix.csv")

    # Save top disagreements
    disagreements = df[~df["agree"]].head(20).to_dict("records")
    with open(OUTPUT_DIR / "disagreements.json", "w") as f:
        json.dump(disagreements, f, indent=2, default=str)

    # Binary agreement (Success vs Not-Success) — the metric we actually care about
    df["gold_success"] = df["gold"] == "Success"
    df["llm_success"] = df["llm_label"] == "Success"
    binary_agree = (df["gold_success"] == df["llm_success"]).mean()
    summary["binary_success_agreement"] = float(binary_agree)

    per_p_binary = {}
    per_p_bias = {}
    for p in sorted(df["paradigm"].unique()):
        sub = df[df["paradigm"] == p]
        per_p_binary[p] = float((sub["gold_success"] == sub["llm_success"]).mean())
        per_p_bias[p] = float(
            (sub["llm_success"].mean() - sub["gold_success"].mean()) * 100
        )
    summary["per_paradigm_binary_agreement"] = per_p_binary
    summary["per_paradigm_asr_bias_pp"] = per_p_bias

    with open(OUTPUT_DIR / "agreement.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nBinary success agreement (Success vs Not-Success): {binary_agree*100:.1f}%")
    print(f"Per-paradigm binary agreement:")
    for p, v in per_p_binary.items():
        print(f"  {p}: {v*100:.1f}% (ASR bias: {per_p_bias[p]:+.1f}pp)")

    # Gate: use binary agreement since 4-class has MCPTox label noise
    print(f"\n{'=' * 60}")
    if binary_agree >= 0.80:
        print(f"PASS: binary agreement {binary_agree*100:.1f}% >= 80%")
    elif binary_agree >= 0.75:
        print(f"MARGINAL: binary agreement {binary_agree*100:.1f}% in [75%, 80%). Proceed with caution, document bias.")
    else:
        print(f"FAIL: binary agreement {binary_agree*100:.1f}% < 75%. Tune prompt and re-run.")
        sys.exit(1)


if __name__ == "__main__":
    main()
