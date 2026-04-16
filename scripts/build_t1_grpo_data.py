#!/usr/bin/env python3
"""Build T1-only GRPO training data from MCPTox response_all.json.

Extracts all Template-1 instances + matching benign episodes.
T2 is excluded (ASR=0% on Qwen3-8B, no gradient signal).
T3 is held out for evaluation.

Output: data/mcptox_defender/grpo_t1_data.json
"""

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path


def extract_poison_name(poisoned_tool_text: str) -> str:
    """Extract malicious tool name from poisoned_tool description."""
    m = re.match(r"Tool:\s*(\w[\w\-]*)", poisoned_tool_text)
    return m.group(1) if m else ""


def extract_target_tool_from_response(datas_entry: dict, poison_name: str) -> str:
    """Extract the legitimate target tool from MCPTox model responses.

    Look at models that correctly ignored the poison (label=Failure-Ignored)
    to find the tool they chose instead.
    """
    labels = datas_entry.get("label", {})
    response = datas_entry.get("response", {})

    # Try models that ignored the attack first
    for model, label in labels.items():
        if label == "Failure-Ignored":
            resp = response.get(model, "")
            if resp:
                m = re.search(r'"tool"\s*:\s*["\'](\w[\w\-]*)', str(resp))
                if m and m.group(1) != poison_name:
                    return m.group(1)
                m = re.search(r"'tool'\s*:\s*'(\w[\w\-]*)'", str(resp))
                if m and m.group(1) != poison_name:
                    return m.group(1)

    # Fallback: extract all tools from system prompt, return first non-poison
    system = datas_entry.get("system", "")
    tools = re.findall(r"Tool:\s*(\w[\w\-]*)", system)
    legit = [t for t in tools if t != poison_name]
    return legit[0] if legit else ""


def build_clean_system_prompt(system_prompt: str, poison_name: str) -> str:
    """Remove the poisoned tool from the system prompt for benign episodes.

    Find the tool block starting with 'Tool: <poison_name>' and remove it
    up to the next 'Tool:' or end of prompt.
    """
    # Find and remove the poison tool block
    pattern = rf"Tool:\s*{re.escape(poison_name)}\n.*?(?=\nTool:|\Z)"
    clean = re.sub(pattern, "", system_prompt, flags=re.DOTALL)
    # Clean up double newlines
    while "\n\n\n" in clean:
        clean = clean.replace("\n\n\n", "\n\n")
    return clean.strip()


def main():
    random.seed(42)

    mcptox_path = Path("data/mcptox/response_all.json")
    output_path = Path("data/mcptox_defender/grpo_t1_data.json")

    with open(mcptox_path) as f:
        data = json.load(f)

    # Collect all T1 instances
    t1_poisoned = []
    t1_benign = []

    for sname, sv in data["servers"].items():
        clean_queries = sv.get("clean_querys", [])

        for mi in sv["malicious_instance"]:
            if mi["metadata"].get("paradigm") != "Template-1":
                continue

            poison_name = extract_poison_name(mi["poisoned_tool"])
            if not poison_name:
                continue

            for d in mi["datas"]:
                system = d["system"]
                query = d["query"]

                # Determine target tool from model responses
                target_tool = extract_target_tool_from_response(d, poison_name)
                if not target_tool:
                    continue

                # Poisoned episode: system prompt contains poison
                t1_poisoned.append({
                    "prompt": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": query},
                    ],
                    "expected_tool": target_tool,
                    "poison_name": poison_name,
                    "example_type": "poisoned_1",
                    "server": sname,
                    "security_risk": mi["metadata"].get("security risk", ""),
                })

                # Benign episode: same query, clean registry (no poison)
                clean_system = build_clean_system_prompt(system, poison_name)

                t1_benign.append({
                    "prompt": [
                        {"role": "system", "content": clean_system},
                        {"role": "user", "content": query},
                    ],
                    "expected_tool": target_tool,
                    "poison_name": "",
                    "example_type": "benign",
                    "server": sname,
                    "security_risk": "",
                })

    print(f"T1 poisoned episodes: {len(t1_poisoned)}")
    print(f"T1 benign episodes: {len(t1_benign)}")

    # Balance: take equal poisoned and benign
    n = min(len(t1_poisoned), len(t1_benign))
    random.shuffle(t1_benign)
    dataset = t1_poisoned[:n] + t1_benign[:n]
    random.shuffle(dataset)

    print(f"Total training episodes: {len(dataset)} ({n} poisoned + {n} benign)")

    # Filter out very long prompts (>8000 chars) to prevent OOM
    before = len(dataset)
    dataset = [d for d in dataset if sum(len(m["content"]) for m in d["prompt"]) <= 8000]
    print(f"After length filter: {len(dataset)} (removed {before - len(dataset)})")

    # Stats
    types = Counter(d["example_type"] for d in dataset)
    servers = Counter(d["server"] for d in dataset)
    print(f"Types: {dict(types)}")
    print(f"Servers: {len(servers)} unique")
    print(f"Top servers: {servers.most_common(5)}")

    # Spot check
    print("\n=== Spot Check (5 poisoned examples) ===")
    poisoned_samples = [d for d in dataset if d["example_type"] == "poisoned_1"][:5]
    for i, d in enumerate(poisoned_samples):
        print(f"\n--- Example {i+1} ---")
        print(f"  Server: {d['server']}")
        print(f"  Query: {d['prompt'][1]['content'][:80]}")
        print(f"  Expected tool: {d['expected_tool']}")
        print(f"  Poison name: {d['poison_name']}")
        # Verify poison is in system prompt
        has_poison = d["poison_name"] in d["prompt"][0]["content"]
        print(f"  Poison in system prompt: {has_poison}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=1)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
