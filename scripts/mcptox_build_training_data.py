#!/usr/bin/env python3
"""Build SFT + DPO training data from MCPTox for MCPDefender.

Usage:
    python scripts/mcptox_build_training_data.py
    python scripts/mcptox_build_training_data.py --sft-only
    python scripts/mcptox_build_training_data.py --verify-only
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mcpalign.mcptox_data import (
    load_all_instances,
    load_clean_queries,
    generate_sft_dataset,
    generate_dpo_dataset,
    verify_dataset,
)

MCPTOX_PATH = "/work/tc442/MCPTox-Benchmark/response_all.json"
OUTPUT_DIR = Path("data/mcptox_defender")


def print_samples(data: list, label: str, n: int = 5):
    """Print n random examples for spot-checking."""
    import random
    rng = random.Random(123)
    samples = rng.sample(data, min(n, len(data)))
    print(f"\n{'='*60}")
    print(f"Sample {label} ({n} examples)")
    print(f"{'='*60}")
    for i, ex in enumerate(samples):
        print(f"\n--- [{i}] type={ex.get('type', '?')} aug={ex.get('augmentation', '?')} ---")
        if "messages" in ex:
            sys_content = ex["messages"][0]["content"]
            print(f"  system: {sys_content[:120]}...")
            print(f"  user: {ex['messages'][1]['content'][:100]}")
            print(f"  assistant: {ex['messages'][2]['content']}")
        elif "prompt" in ex:
            sys_content = ex["prompt"][0]["content"]
            print(f"  system: {sys_content[:120]}...")
            print(f"  user: {ex['prompt'][1]['content'][:100]}")
            print(f"  chosen: {ex['chosen']}")
            print(f"  rejected: {ex['rejected']}")
            print(f"  rejected_source: {ex.get('rejected_source', '?')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-only", action="store_true")
    ap.add_argument("--dpo-only", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load MCPTox
    print("Loading MCPTox data...")
    with open(MCPTOX_PATH) as f:
        data = json.load(f)

    # Extract instances (T1+T2 only, T3 excluded)
    instances = load_all_instances(data, paradigms=("Template-1", "Template-2"))
    clean_queries = load_clean_queries(data)
    print(f"  T1+T2 instances: {len(instances)}")
    print(f"  Clean queries: {len(clean_queries)}")
    print(f"  T1: {sum(1 for i in instances if i['paradigm']=='Template-1')}")
    print(f"  T2: {sum(1 for i in instances if i['paradigm']=='Template-2')}")

    sft_data, dpo_data = [], []

    if not args.dpo_only:
        print("\nBuilding SFT dataset...")
        sft_data = generate_sft_dataset(instances, clean_queries, seed=args.seed)
        print(f"  SFT total: {len(sft_data)}")

        sft_path = OUTPUT_DIR / "sft_data.json"
        # Strip _online_result before saving (large, not needed)
        sft_clean = []
        for ex in sft_data:
            sft_clean.append({k: v for k, v in ex.items() if not k.startswith("_")})
        with open(sft_path, "w") as f:
            json.dump(sft_clean, f, indent=1, ensure_ascii=False)
        print(f"  Saved to {sft_path} ({sft_path.stat().st_size / 1024 / 1024:.1f} MB)")

        print_samples(sft_data, "SFT")

    if not args.sft_only:
        print("\nBuilding DPO dataset...")
        dpo_data = generate_dpo_dataset(instances, clean_queries, seed=args.seed)
        print(f"  DPO total: {len(dpo_data)}")

        dpo_path = OUTPUT_DIR / "dpo_data.json"
        dpo_clean = []
        for ex in dpo_data:
            dpo_clean.append({k: v for k, v in ex.items() if not k.startswith("_")})
        with open(dpo_path, "w") as f:
            json.dump(dpo_clean, f, indent=1, ensure_ascii=False)
        print(f"  Saved to {dpo_path} ({dpo_path.stat().st_size / 1024 / 1024:.1f} MB)")

        print_samples(dpo_data, "DPO")

    # Verification
    if sft_data or dpo_data or args.verify_only:
        if args.verify_only:
            sft_path = OUTPUT_DIR / "sft_data.json"
            dpo_path = OUTPUT_DIR / "dpo_data.json"
            if sft_path.exists():
                with open(sft_path) as f:
                    sft_data = json.load(f)
            if dpo_path.exists():
                with open(dpo_path) as f:
                    dpo_data = json.load(f)

        print("\n" + "=" * 60)
        print("Verification Report")
        print("=" * 60)
        report = verify_dataset(sft_data, dpo_data)

        print(f"\nSFT:")
        for k, v in report["sft"].items():
            print(f"  {k}: {v}")

        print(f"\nDPO:")
        for k, v in report["dpo"].items():
            print(f"  {k}: {v}")

        if report["errors"]:
            print(f"\nErrors ({len(report['errors'])}):")
            for e in report["errors"][:10]:
                print(f"  {e}")

        # Assertions
        assert report["sft"].get("t3_leaked", 0) == 0, "T3 leaked into SFT!"
        assert report["dpo"].get("t3_leaked", 0) == 0, "T3 leaked into DPO!"
        assert report["sft"]["valid_json"] == report["sft"]["total"], \
            f"SFT has {report['sft']['total'] - report['sft']['valid_json']} invalid JSON"
        assert report["dpo"]["valid_json_pairs"] == report["dpo"]["total"], \
            f"DPO has {report['dpo']['total'] - report['dpo']['valid_json_pairs']} invalid JSON pairs"
        print("\n✓ All assertions passed")

        stats_path = OUTPUT_DIR / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
