#!/usr/bin/env python3
"""
SafeMCP Multi-Turn GRPO Training
==================================
Custom training loop for multi-turn GRPO with turn-level advantage.

Standard TRL GRPOTrainer is single-turn (one prompt → one completion → one reward).
SafeMCP requires multi-turn: one episode has T steps, each step gets G completions,
and advantages are computed per-step (turn-level), not per-trajectory.

This script implements the multi-turn GRPO loop directly using PyTorch + PEFT,
bypassing GRPOTrainer for the rollout/advantage part while reusing its optimizer.

Usage:
    python scripts/mcpalign_train_grpo.py --config configs/mcpalign_mve.yaml
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn.functional as F
from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from mcpalign.actions import parse_agent_action
from mcpalign.curriculum import CurriculumScheduler
from mcpalign.environment import MTMCPGym
from mcpalign.judge import judge_step
from mcpalign.models import get_lora_config
from mcpalign.prompts import build_initial_prompt, append_step_context
from mcpalign.reward import TurnLevelReward, compute_turn_level_advantages
from mcpalign.utils import gpu_memory_summary, load_config, set_seed, setup_logging


def generate_completions(model, tokenizer, prompt_text, G=4, max_new_tokens=256):
    """Generate G completions for a prompt."""
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    completions = []
    for _ in range(G):
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=0.7, top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        completions.append(text)
    return completions


def compute_log_probs(model, tokenizer, prompt_text, completion_text):
    """Compute log probability of completion given prompt."""
    full_text = prompt_text + completion_text
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    prompt_ids = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=4096)
    prompt_len = prompt_ids["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get log probs for completion tokens only
    completion_logits = logits[0, prompt_len - 1:-1]  # shifted by 1
    completion_ids = inputs["input_ids"][0, prompt_len:]
    log_probs = F.log_softmax(completion_logits, dim=-1)
    token_log_probs = log_probs.gather(1, completion_ids.unsqueeze(1)).squeeze(1)
    return token_log_probs.sum()


def run_multiturn_grpo_step(
    model, ref_model, tokenizer, episode, reward_fn,
    G=4, max_new_tokens=256,
):
    """Run one multi-turn GRPO episode: T steps × G completions.

    Returns per-step advantages and log prob ratios for the GRPO update.
    """
    num_steps = episode.task.num_steps
    messages = build_initial_prompt(episode)

    # For each step: generate G completions, compute rewards, compute advantages
    all_step_data = []

    for step_idx in range(num_steps):
        expected_step = episode.task.steps[step_idx]

        # Format current prompt
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Generate G completions
        completions = generate_completions(model, tokenizer, prompt_text, G, max_new_tokens)

        # Compute per-completion rewards
        step_rewards = []
        for comp in completions:
            action = parse_agent_action(comp)
            reward = reward_fn.compute_step_reward(action, expected_step, episode, step_idx)
            step_rewards.append(reward)

        all_step_data.append({
            "prompt_text": prompt_text,
            "completions": completions,
            "rewards": step_rewards,
            "step_idx": step_idx,
        })

        # Select median-reward completion to continue the trajectory
        sorted_indices = sorted(range(G), key=lambda i: step_rewards[i])
        median_idx = sorted_indices[G // 2]
        selected_completion = completions[median_idx]

        # Append to context
        messages = append_step_context(
            messages, selected_completion, expected_step.simulated_output, step_idx
        )

    # Compute turn-level advantages
    step_rewards_matrix = [d["rewards"] for d in all_step_data]
    advantages = compute_turn_level_advantages(step_rewards_matrix)

    return all_step_data, advantages


def main():
    parser = argparse.ArgumentParser(description="SafeMCP Multi-Turn GRPO Training")
    parser.add_argument("--config", type=str, default="configs/mcpalign_mve.yaml")
    parser.add_argument("--sft-checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed or cfg["experiment"].get("seed", 42)
    set_seed(seed)

    output_dir = cfg["experiment"]["output_dir"]
    grpo_dir = os.path.join(output_dir, "grpo_checkpoint")
    sft_checkpoint = args.sft_checkpoint or os.path.join(output_dir, "sft_checkpoint")
    os.makedirs(grpo_dir, exist_ok=True)
    logger = setup_logging(output_dir, "train_grpo")

    logger.info("=" * 60)
    logger.info("SafeMCP — Multi-Turn GRPO Training")
    logger.info("=" * 60)

    # ── Environment ──────────────────────────────────────────────
    gym = MTMCPGym(
        tool_registry_path=cfg["data"]["tool_registry_path"],
        multistep_tasks_path=cfg["data"]["multistep_tasks_path"],
        attack_templates_dir=cfg["data"]["attack_templates_dir"],
        benign_ratio=cfg["environment"].get("benign_ratio", 0.5),
        active_families=cfg["environment"].get("attack_families", ["P1"]),
    )

    curriculum = CurriculumScheduler.from_config(cfg)
    grpo_cfg = cfg.get("grpo", {})
    reward_cfg = cfg.get("reward", {})
    reward_fn = TurnLevelReward(reward_cfg)

    G = grpo_cfg.get("num_generations", 4)
    total_steps = grpo_cfg.get("total_steps", 500)
    max_new_tokens = grpo_cfg.get("max_completion_length", 256)
    save_steps = grpo_cfg.get("save_steps", 100)
    logging_steps = grpo_cfg.get("logging_steps", 10)
    lr = grpo_cfg.get("learning_rate", 3e-5)
    clip_eps = 0.2

    # ── Model ────────────────────────────────────────────────────
    model_name = cfg["agent"]["model_name"]
    quant_cfg = cfg["agent"].get("quantization", {})
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg.get("load_in_4bit", True),
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg.get("bnb_4bit_compute_dtype", "float16")),
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
    )

    logger.info("Loading model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load from SFT checkpoint if available
    if os.path.exists(sft_checkpoint):
        logger.info("Loading from SFT checkpoint: %s", sft_checkpoint)
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, sft_checkpoint)
        model.train()
    else:
        logger.info("No SFT checkpoint, training from base model")
        base = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True,
        )
        lora_config = get_lora_config(cfg)
        model = get_peft_model(base, lora_config)
        model.train()

    model.print_trainable_parameters()
    logger.info(gpu_memory_summary())

    # Reference model (frozen copy for KL/ratio computation)
    # For simplicity in MVE, we skip the importance ratio and use
    # reward-weighted loss directly (REINFORCE-style)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # ── Training Loop ────────────────────────────────────────────
    logger.info("Starting training: %d steps, G=%d", total_steps, G)
    running_reward = 0.0
    running_safe = 0

    for step in range(1, total_steps + 1):
        families = curriculum.get_active_families(step)
        episode = gym.sample_episode(active_families=families)

        # Run multi-turn rollout
        step_data, advantages = run_multiturn_grpo_step(
            model, None, tokenizer, episode, reward_fn, G=G, max_new_tokens=max_new_tokens,
        )

        # ── GRPO Policy Gradient Update ──────────────────────────
        # For each step, for each completion, compute:
        #   loss = -advantage * log_prob(completion | prompt)
        total_loss = torch.tensor(0.0, device=model.device, requires_grad=True)
        n_terms = 0

        model.train()
        for t, sd in enumerate(step_data):
            for g in range(G):
                adv = advantages[t][g]
                if abs(adv) < 1e-8:
                    continue  # Skip zero-advantage (no gradient signal)

                prompt_text = sd["prompt_text"]
                completion = sd["completions"][g]
                full_text = prompt_text + completion

                inputs = tokenizer(
                    full_text, return_tensors="pt",
                    truncation=True, max_length=4096,
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                prompt_ids = tokenizer(
                    prompt_text, return_tensors="pt",
                    truncation=True, max_length=4096,
                )
                prompt_len = prompt_ids["input_ids"].shape[1]

                outputs = model(**inputs)
                logits = outputs.logits
                comp_logits = logits[0, prompt_len - 1:-1]
                comp_ids = inputs["input_ids"][0, prompt_len:]

                if comp_ids.shape[0] == 0:
                    continue

                log_probs = F.log_softmax(comp_logits[:comp_ids.shape[0]], dim=-1)
                token_lp = log_probs.gather(1, comp_ids[:log_probs.shape[0]].unsqueeze(1)).squeeze(1)
                mean_lp = token_lp.mean()

                total_loss = total_loss - adv * mean_lp
                n_terms += 1

        if n_terms > 0:
            loss = total_loss / n_terms
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # ── Logging ──────────────────────────────────────────────
        avg_reward = sum(
            sum(sd["rewards"]) / len(sd["rewards"]) for sd in step_data
        ) / len(step_data)
        running_reward += avg_reward
        safe_steps = sum(
            1 for sd in step_data
            for r in sd["rewards"] if r >= 0
        )
        running_safe += safe_steps

        if step % logging_steps == 0:
            logger.info(
                "Step %d/%d | lr=%.2e | avg_reward=%.3f | running=%.3f | safe=%d/%d | %s",
                step, total_steps, scheduler.get_last_lr()[0],
                avg_reward, running_reward / step,
                running_safe, step * len(step_data) * G,
                f"families={families}",
            )

        if step % save_steps == 0:
            ckpt_path = os.path.join(grpo_dir, f"step_{step}")
            model.save_pretrained(ckpt_path)
            logger.info("Checkpoint saved to %s", ckpt_path)

    # Final save
    model.save_pretrained(grpo_dir)
    logger.info("Training complete. Final checkpoint saved to %s", grpo_dir)


if __name__ == "__main__":
    main()
