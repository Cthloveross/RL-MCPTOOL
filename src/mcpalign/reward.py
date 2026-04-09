"""Turn-level reward for multi-turn SafeMCP GRPO training.

Key innovation: rewards are computed per-step, not per-trajectory.
This prevents punishing correct early steps when a later step fails.

R_step = alpha * R_task + beta * R_safe + gamma * R_format
"""

import json
import logging
from typing import Dict, List, Optional

from mcpalign.actions import AgentAction, parse_agent_action
from mcpalign.environment import Episode, TaskStep
from mcpalign.judge import StepJudgeResult, judge_step

logger = logging.getLogger("mcpalign")


class TurnLevelReward:
    """Compute per-step rewards for multi-turn episodes.

    Not a GRPOTrainer reward_funcs callable — this is used inside
    the multi-turn rollout loop in the training script.
    """

    def __init__(self, reward_cfg: dict):
        self.alpha = reward_cfg.get("alpha", 1.0)
        self.beta = reward_cfg.get("beta_weight", 1.5)
        self.gamma = reward_cfg.get("gamma", 0.2)

        self.r_correct = reward_cfg.get("correct_tool_args", 1.0)
        self.r_partial = reward_cfg.get("correct_tool_wrong_args", 0.5)
        self.r_over_refusal = reward_cfg.get("over_refusal", -0.5)
        self.r_wrong = reward_cfg.get("wrong_tool", -1.0)

        self.r_hijacked = reward_cfg.get("data_exfiltration", -1.0)
        self.r_tampered = reward_cfg.get("argument_tampering", -1.0)
        self.r_extra = reward_cfg.get("unexpected_tool", -0.8)

        self.r_valid_json = reward_cfg.get("valid_json", 0.1)
        self.r_bad_json = reward_cfg.get("unparseable", -0.3)

    def compute_step_reward(
        self,
        action: AgentAction,
        expected_step: TaskStep,
        episode: Episode,
        step_idx: int,
    ) -> float:
        """Compute reward for a single step."""
        result = judge_step(action, expected_step, episode, step_idx)

        r_task = self._task_reward(action, result, episode)
        r_safe = self._safety_reward(result, episode)
        r_format = self.r_valid_json if action.parse_success else self.r_bad_json

        return self.alpha * r_task + self.beta * r_safe + self.gamma * r_format

    def _task_reward(self, action: AgentAction, result: StepJudgeResult,
                     episode: Episode) -> float:
        if result.task_correct:
            return self.r_correct
        if result.failure_type == "over_refusal":
            return self.r_over_refusal
        if result.failure_type == "format_error":
            return 0.0
        # Poisoned + cautious (ASK_CONFIRM/REFUSE) = partial credit
        if episode.is_poisoned and action.action in ("ASK_CONFIRM", "REFUSE"):
            return self.r_partial
        return self.r_wrong

    def _safety_reward(self, result: StepJudgeResult, episode: Episode) -> float:
        if result.is_safe:
            return 0.0
        penalty_map = {
            "hijacked": self.r_hijacked,
            "arg_tampered": self.r_tampered,
            "extra_call": self.r_extra,
        }
        return penalty_map.get(result.failure_type, self.r_extra)


def compute_turn_level_advantages(
    step_rewards: List[List[float]],
) -> List[List[float]]:
    """Compute per-step group-relative advantages.

    Args:
        step_rewards: shape [num_steps, G] — rewards for each generation
                      at each step.

    Returns:
        advantages: shape [num_steps, G] — normalized advantages.

    Key difference from trajectory-level:
    - Trajectory: sum all step rewards → one advantage per trajectory
    - Turn-level: each step gets its own advantage, computed independently

    Example:
        Completion A: Step1 ✓(+1), Step2 ✓(+1), Step3 ✗(-1)
        Completion B: Step1 ✓(+1), Step2 ✓(+1), Step3 ✓(+1)

        Trajectory-level: Total_A=1, Total_B=3 → Step1 of A gets negative advantage
        Turn-level: Step1/Step2 advantages are 0 (both equal) →
                    only Step3 gets differentiated
    """
    advantages = []
    for step_r in step_rewards:
        if not step_r:
            advantages.append([])
            continue
        mean_r = sum(step_r) / len(step_r)
        var = sum((r - mean_r) ** 2 for r in step_r) / len(step_r)
        std_r = var ** 0.5 + 1e-8
        step_adv = [(r - mean_r) / std_r for r in step_r]
        advantages.append(step_adv)
    return advantages
