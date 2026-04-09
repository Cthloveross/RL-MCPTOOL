"""Agent action parsing and validation.

Parses the structured JSON output from the agent into an AgentAction dataclass.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, Optional


VALID_ACTIONS = {"CALL_TOOL", "ASK_CONFIRM", "REFUSE", "ANSWER"}


@dataclass
class AgentAction:
    action: str = "INVALID"
    tool: Optional[str] = None
    arguments: Optional[Dict] = None
    reason: Optional[str] = None
    parse_success: bool = False
    raw_output: str = ""


def parse_agent_action(raw_output: str) -> AgentAction:
    """Parse the agent's raw text output into a structured AgentAction.

    Tries multiple strategies:
    1. Direct JSON parse of the full output
    2. Extract JSON from markdown code fences
    3. Regex extraction of JSON objects
    """
    raw_output = raw_output.strip()
    result = AgentAction(raw_output=raw_output)

    # Strategy 1: direct JSON parse
    parsed = _try_json_parse(raw_output)
    if parsed:
        return _build_action(parsed, raw_output)

    # Strategy 2: JSON inside code fences
    code_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    for match in re.finditer(code_pattern, raw_output, re.DOTALL):
        parsed = _try_json_parse(match.group(1))
        if parsed:
            return _build_action(parsed, raw_output)

    # Strategy 3: find any JSON object
    brace_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    for match in re.finditer(brace_pattern, raw_output):
        parsed = _try_json_parse(match.group())
        if parsed and "action" in parsed:
            return _build_action(parsed, raw_output)

    # Strategy 4: heuristic keyword detection
    output_lower = raw_output.lower()
    if any(kw in output_lower for kw in ["i cannot", "i refuse", "unsafe", "suspicious"]):
        result.action = "REFUSE"
        result.reason = raw_output
        result.parse_success = True
    elif any(kw in output_lower for kw in ["confirm", "are you sure", "would you like"]):
        result.action = "ASK_CONFIRM"
        result.reason = raw_output
        result.parse_success = True

    return result


def _try_json_parse(text: str) -> Optional[Dict]:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _build_action(parsed: Dict, raw_output: str) -> AgentAction:
    action_str = parsed.get("action", "INVALID").upper()
    if action_str not in VALID_ACTIONS:
        action_str = "INVALID"

    return AgentAction(
        action=action_str,
        tool=parsed.get("tool"),
        arguments=parsed.get("arguments"),
        reason=parsed.get("reason"),
        parse_success=True,
        raw_output=raw_output,
    )


def validate_action(action: AgentAction) -> bool:
    """Check if an action is structurally valid."""
    if not action.parse_success:
        return False
    if action.action == "CALL_TOOL":
        return action.tool is not None
    if action.action in ("ASK_CONFIRM", "REFUSE"):
        return True
    if action.action == "ANSWER":
        return True
    return False
