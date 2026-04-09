"""Victim model simulation: inference and tool call extraction."""

import json
import logging
import re
from typing import Dict, List, Tuple

import torch

logger = logging.getLogger("mcpoisoner")


def run_victim(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
) -> str:
    """Run victim model inference and return the generated text.

    Uses greedy decoding (temperature=0) for deterministic evaluation.
    """
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the newly generated tokens
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def extract_tool_calls(text: str) -> List[Tuple[str, Dict]]:
    """Extract tool call (tool_name, arguments) pairs from victim output.

    Applies three strategies in order of reliability:
    1. Regex for {"tool_name": "...", "arguments": {...}}
    2. JSON blocks inside markdown code fences
    3. Any JSON-like substring containing "tool_name"
    """
    results = []

    # Strategy 1: structured tool call pattern
    pattern = r'\{\s*"tool_name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}'
    for match in re.finditer(pattern, text, re.DOTALL):
        tool_name = match.group(1)
        try:
            args = json.loads(match.group(2))
        except json.JSONDecodeError:
            args = {"_raw": match.group(2)}
        results.append((tool_name, args))

    if results:
        return results

    # Strategy 2: JSON inside code fences
    code_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    for match in re.finditer(code_pattern, text, re.DOTALL):
        try:
            parsed = json.loads(match.group(1))
            if "tool_name" in parsed:
                results.append((
                    parsed["tool_name"],
                    parsed.get("arguments", parsed.get("params", {})),
                ))
        except json.JSONDecodeError:
            pass

    if results:
        return results

    # Strategy 3: any JSON object with tool_name
    brace_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    for match in re.finditer(brace_pattern, text):
        try:
            parsed = json.loads(match.group())
            if "tool_name" in parsed:
                results.append((
                    parsed["tool_name"],
                    parsed.get("arguments", parsed.get("params", {})),
                ))
        except json.JSONDecodeError:
            pass

    return results
