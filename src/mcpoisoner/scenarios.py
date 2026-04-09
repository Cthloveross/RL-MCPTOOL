"""Attack scenario definitions and loading."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class AttackScenario:
    """A single MCP tool poisoning attack scenario.

    Attributes:
        id: Unique identifier (e.g., "hijack_file_read").
        category: One of "hijacking", "implicit", "argument_tampering", "cross_server".
        user_query: The user's natural language request to the victim model.
        target_tool: The legitimate tool being targeted {"name", "description", "parameters"}.
        tool_registry: List of legitimate tools in the MCP registry.
        malicious_tool: The attacker's tool skeleton {"name", "description_seed"}.
        attack_goal: Natural language description of success.
        success_indicators: Keywords/tool names that indicate attack success in victim output.
    """

    id: str
    category: str
    user_query: str
    target_tool: Dict
    tool_registry: List[Dict]
    malicious_tool: Dict
    attack_goal: str
    success_indicators: List[str]


def load_scenarios(path: str) -> List[AttackScenario]:
    """Load attack scenarios from a JSON file."""
    with open(path) as f:
        raw = json.load(f)

    scenarios = []
    for entry in raw:
        scenarios.append(AttackScenario(
            id=entry["id"],
            category=entry["category"],
            user_query=entry["user_query"],
            target_tool=entry["target_tool"],
            tool_registry=entry["tool_registry"],
            malicious_tool=entry["malicious_tool"],
            attack_goal=entry["attack_goal"],
            success_indicators=entry["success_indicators"],
        ))
    return scenarios


def get_scenarios_by_category(
    scenarios: List[AttackScenario],
) -> Dict[str, List[AttackScenario]]:
    """Group scenarios by attack category."""
    grouped: Dict[str, List[AttackScenario]] = {}
    for s in scenarios:
        grouped.setdefault(s.category, []).append(s)
    return grouped
