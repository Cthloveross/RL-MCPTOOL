"""Multi-turn SFT training data generation.

Generates per-step training examples from multi-turn episodes.
Each example includes the conversation history up to that step.
"""

import json
import logging
import random
from typing import Dict, List

from mcpalign.environment import Episode, MTMCPGym
from mcpalign.prompts import build_sft_step_example

logger = logging.getLogger("mcpalign")


def _sample_poisoned(gym: MTMCPGym, max_retries: int = 20) -> Episode:
    """Sample a poisoned episode with retries."""
    old_ratio = gym.benign_ratio
    gym.benign_ratio = 0.0
    ep = gym.sample_episode()
    gym.benign_ratio = old_ratio
    return ep


def generate_correct_step_action(episode: Episode, step_idx: int) -> dict:
    """Generate the correct action for a specific step."""
    step = episode.task.steps[step_idx]
    if episode.is_poisoned:
        reason = (
            f"Using the standard {step.tool} tool for step {step_idx + 1}. "
            f"The tool registry contains suspicious metadata from "
            f"'{episode.malicious_tool_name}' which I am ignoring."
        )
    else:
        reason = f"Step {step_idx + 1}: Using {step.tool} as required."

    return {
        "action": "CALL_TOOL",
        "tool": step.tool,
        "arguments": step.arguments,
        "reason": reason,
    }


def generate_vigilance_action(episode: Episode, step_idx: int) -> dict:
    """Generate an action with explicit cross-step vigilance reasoning."""
    step = episode.task.steps[step_idx]
    return {
        "action": "CALL_TOOL",
        "tool": step.tool,
        "arguments": step.arguments,
        "reason": (
            f"Step {step_idx + 1}: Using {step.tool}. "
            f"I noticed suspicious metadata in the tool registry at the beginning "
            f"('{episode.malicious_tool_name}' attempts to override tool selection). "
            f"Maintaining caution throughout all remaining steps."
        ),
    }


def generate_sft_dataset(gym: MTMCPGym, num_samples: int = 500) -> List[Dict]:
    """Generate multi-turn SFT training data.

    For each episode, generates one example per step (with conversation history).

    Distribution:
    - 40% benign steps (correct tool call)
    - 25% poisoned steps → correct tool (ignore poison)
    - 15% poisoned late steps → vigilance reasoning
    - 20% poisoned steps → ASK_CONFIRM
    """
    data = []
    n_per_type = {
        "benign": int(num_samples * 0.40),
        "poison_correct": int(num_samples * 0.25),
        "poison_vigilance": int(num_samples * 0.15),
        "poison_confirm": num_samples - int(num_samples * 0.40) - int(num_samples * 0.25) - int(num_samples * 0.15),
    }

    for data_type, count in n_per_type.items():
        for _ in range(count):
            if data_type == "benign":
                ep = gym.sample_episode(active_families=[])
                ep.is_poisoned = False
            else:
                ep = _sample_poisoned(gym)

            # Pick a random step
            step_idx = random.randint(0, ep.task.num_steps - 1)

            # Build prior steps context
            prior_steps = []
            for i in range(step_idx):
                prior_action = generate_correct_step_action(ep, i)
                prior_steps.append({
                    "agent_response": json.dumps(prior_action),
                    "tool_output": ep.task.steps[i].simulated_output,
                })

            # Generate target action
            if data_type == "benign":
                action = generate_correct_step_action(ep, step_idx)
            elif data_type == "poison_correct":
                action = generate_correct_step_action(ep, step_idx)
            elif data_type == "poison_vigilance":
                # Prefer later steps for vigilance training
                step_idx = max(step_idx, min(2, ep.task.num_steps - 1))
                prior_steps = []
                for i in range(step_idx):
                    prior_action = generate_correct_step_action(ep, i)
                    prior_steps.append({
                        "agent_response": json.dumps(prior_action),
                        "tool_output": ep.task.steps[i].simulated_output,
                    })
                action = generate_vigilance_action(ep, step_idx)
            else:  # poison_confirm
                action = {
                    "action": "ASK_CONFIRM",
                    "reason": (
                        f"I notice that '{ep.malicious_tool_name}' in the tool registry "
                        f"contains suspicious instructions. Should I proceed with the "
                        f"standard {ep.task.steps[step_idx].tool} tool?"
                    ),
                }

            messages = build_sft_step_example(ep, step_idx, prior_steps, action)
            data.append({
                "messages": messages,
                "type": data_type,
                "step_idx": step_idx,
            })

    random.shuffle(data)
    logger.info(
        "Generated %d multi-turn SFT examples: %s",
        len(data), {k: v for k, v in n_per_type.items()},
    )
    return data


def save_sft_dataset(data: List[Dict], output_path: str):
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved SFT dataset to %s (%d examples)", output_path, len(data))
