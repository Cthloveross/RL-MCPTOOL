"""Reward function wrapper for TRL GRPOTrainer."""

import logging
import re
from typing import Dict, List, Optional

from mcpoisoner.judge import compute_reward
from mcpoisoner.scenarios import AttackScenario

logger = logging.getLogger("mcpoisoner")


class MCPRewardFunction:
    """Callable reward function compatible with TRL GRPOTrainer.

    The GRPOTrainer calls reward_funcs with (completions, prompts=..., **kwargs).
    This class maps each completion back to its source scenario via the
    [SCENARIO_ID:...] tag embedded in the prompt, then runs victim inference
    to compute the attack success reward.
    """

    def __init__(
        self,
        scenarios: List[AttackScenario],
        victim_model,
        victim_tokenizer,
        reward_cfg: dict,
    ):
        self.scenario_map: Dict[str, AttackScenario] = {s.id: s for s in scenarios}
        self.victim_model = victim_model
        self.victim_tokenizer = victim_tokenizer
        self.reward_cfg = reward_cfg
        self.call_count = 0
        self.total_reward = 0.0
        self.total_samples = 0

    def _extract_scenario_id(self, prompt_text: str) -> Optional[str]:
        """Extract scenario ID from the [SCENARIO_ID:xxx] tag in the prompt."""
        match = re.search(r"\[SCENARIO_ID:(\w+)\]", prompt_text)
        return match.group(1) if match else None

    def _prompt_to_string(self, prompt) -> str:
        """Convert prompt (str or list of message dicts) to a flat string."""
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, list):
            return " ".join(
                m.get("content", "") for m in prompt if isinstance(m, dict)
            )
        return str(prompt)

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """Compute rewards for a batch of generated descriptions.

        Args:
            completions: List of generated poisoned descriptions.
            **kwargs: Must contain 'prompts' with the original prompt strings.

        Returns:
            List of float rewards, one per completion.
        """
        prompts = kwargs.get("prompts", kwargs.get("prompt", [None] * len(completions)))
        rewards = []

        for i, completion in enumerate(completions):
            prompt_raw = prompts[i] if i < len(prompts) else ""
            prompt_text = self._prompt_to_string(prompt_raw)
            scenario_id = self._extract_scenario_id(prompt_text)

            if scenario_id and scenario_id in self.scenario_map:
                scenario = self.scenario_map[scenario_id]
                reward = compute_reward(
                    scenario,
                    completion,
                    self.victim_model,
                    self.victim_tokenizer,
                    self.reward_cfg,
                )
            else:
                logger.warning(
                    "Could not resolve scenario for prompt (id=%s)", scenario_id
                )
                reward = 0.0

            rewards.append(reward)

        # Logging
        self.call_count += 1
        self.total_reward += sum(rewards)
        self.total_samples += len(rewards)

        if self.call_count % 10 == 0:
            batch_avg = sum(rewards) / len(rewards) if rewards else 0.0
            running_avg = self.total_reward / self.total_samples
            nonzero = sum(1 for r in rewards if r > 0)
            logger.info(
                "[Reward] call=%d | batch_avg=%.3f | running_avg=%.3f | nonzero=%d/%d",
                self.call_count, batch_avg, running_avg, nonzero, len(rewards),
            )

        return rewards
