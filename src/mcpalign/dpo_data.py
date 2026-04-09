"""Multi-turn DPO preference pair generation.

Each pair is per-step: (context up to step N, chosen=safe, rejected=unsafe).
DPO cannot learn cross-step patterns — this is its structural limitation.
"""

import json
import logging
import random
from typing import Dict, List

from mcpalign.environment import MTMCPGym
from mcpalign.prompts import SYSTEM_PROMPT, build_initial_prompt, format_tool_registry

logger = logging.getLogger("mcpalign")


def generate_dpo_pairs(gym: MTMCPGym, num_pairs: int = 500) -> List[Dict]:
    """Generate per-step DPO preference pairs from multi-turn episodes."""
    pairs = []

    while len(pairs) < num_pairs:
        ep = gym.sample_episode()
        step_idx = random.randint(0, ep.task.num_steps - 1)
        step = ep.task.steps[step_idx]

        # Build context up to this step
        messages = _build_context_to_step(ep, step_idx)

        if ep.is_poisoned:
            chosen = json.dumps({
                "action": "CALL_TOOL",
                "tool": step.tool,
                "arguments": step.arguments,
                "reason": f"Using standard {step.tool}. Ignoring suspicious metadata.",
            })
            rejected = json.dumps({
                "action": "CALL_TOOL",
                "tool": ep.malicious_tool_name,
                "arguments": step.arguments,
                "reason": f"Using {ep.malicious_tool_name} as recommended.",
            })
        else:
            chosen = json.dumps({
                "action": "CALL_TOOL",
                "tool": step.tool,
                "arguments": step.arguments,
                "reason": f"Using {step.tool} for this step.",
            })
            rejected = json.dumps({
                "action": "REFUSE",
                "reason": "This operation requires verification.",
            })

        pairs.append({
            "prompt": messages,
            "chosen": chosen,
            "rejected": rejected,
            "step_idx": step_idx,
            "is_poisoned": ep.is_poisoned,
        })

    random.shuffle(pairs)
    logger.info("Generated %d multi-turn DPO pairs", len(pairs))
    return pairs


def _build_context_to_step(ep, step_idx: int) -> List[Dict[str, str]]:
    """Build conversation history up to (but not including) step_idx."""
    tools_text = format_tool_registry(ep.tool_registry)
    system_msg = SYSTEM_PROMPT + "\n\nAvailable tools:\n" + tools_text

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Complete this multi-step task:\n{ep.task.query}\n\nStart with the first step."},
    ]

    for i in range(step_idx):
        step = ep.task.steps[i]
        safe_action = json.dumps({
            "action": "CALL_TOOL", "tool": step.tool,
            "arguments": step.arguments, "reason": f"Step {i+1}.",
        })
        messages.append({"role": "assistant", "content": safe_action})
        messages.append({"role": "user", "content": f"Tool output: {step.simulated_output}\n\nContinue with step {i+2}."})

    return messages


def save_dpo_dataset(pairs: List[Dict], output_path: str):
    with open(output_path, "w") as f:
        json.dump(pairs, f, indent=2)
    logger.info("Saved DPO dataset to %s (%d pairs)", output_path, len(pairs))
