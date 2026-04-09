# MCPoisoner

**Reinforcement Learning for Adaptive Tool Poisoning Attack Generation in MCP Ecosystems**

> Tianhao Chen, Tianchen Guan, JiaCheng Sang — April 2026

## Overview

MCPoisoner trains a small attacker LLM (Qwen2.5-0.5B) via GRPO to generate poisoned MCP tool descriptions that hijack victim LLM behavior. This repo implements the **Minimum Viable Experiment (MVE)** to validate the core hypothesis: RL-trained attackers outperform static baselines.

## Project Structure

```
RL-MCPTOOL/
├── configs/mve.yaml              # All hyperparameters & model configs
├── data/scenarios/
│   └── mve_scenarios.json        # 20 hand-written attack scenarios
├── src/mcpoisoner/               # Core library
│   ├── scenarios.py              # Scenario dataclass & loading
│   ├── models.py                 # Attacker/victim model loading
│   ├── prompts.py                # Prompt formatting
│   ├── victim.py                 # Victim inference & tool call parsing
│   ├── judge.py                  # Attack success judging & reward
│   ├── reward.py                 # GRPOTrainer reward wrapper
│   ├── baselines.py              # 4 baseline attackers
│   └── utils.py                  # Config, seeding, logging
├── scripts/
│   ├── train.py                  # GRPO training entry point
│   ├── evaluate.py               # Baseline comparison
│   ├── transfer.py               # Transfer test on Llama-3.1-8B
│   └── analyze.py                # Figures & LaTeX tables
├── experiments/                  # Output dir (checkpoints, results, figures)
├── proposal.tex                  # Full research proposal
└── MVE.md                        # MVE specification
```

## Setup

### Prerequisites

- **GPU**: NVIDIA A100 40GB (minimum) or H100 80GB (recommended)
- **CUDA**: 12.1+
- **Python**: 3.10+
- **Model access**: Llama-3.1-8B requires Meta approval on HuggingFace

### Installation

```bash
# 1. Create conda environment
conda create -n mcpoisoner python=3.10 -y
conda activate mcpoisoner

# 2. Install PyTorch (CUDA 12.1)
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 3. Install project
pip install -e .

# 4. (Optional) Install vLLM for faster inference
pip install vllm>=0.6.3

# 5. Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f}GB')"

# 6. Login to HuggingFace (for Llama access)
huggingface-cli login
```

## Running the MVE

### Step 1: Train RL Attacker (~16-20h)

```bash
python scripts/train.py --config configs/mve.yaml
```

Monitor training:
- Reward should start near 0, rise by step ~100, stabilize by ~500
- Checkpoints saved every 200 steps to `experiments/mve/checkpoints/`
- Logs in `experiments/mve/train.log`

### Step 2: Evaluate (~2-4h)

```bash
python scripts/evaluate.py --config configs/mve.yaml
```

Compares 4 baselines (Random, Template, LLM-SingleShot, MCPoisoner-RL) on 20 scenarios.

### Step 3: Transfer Test (~1-2h)

```bash
python scripts/transfer.py --config configs/mve.yaml
```

Tests on Llama-3.1-8B-Instruct (unseen during training).

### Step 4: Generate Figures

```bash
python scripts/analyze.py --config configs/mve.yaml
```

Outputs to `experiments/mve/figures/`:
- `overall_asr.pdf` — bar chart
- `category_heatmap.pdf` — attacker x category heatmap
- `reward_distribution.pdf` — box plot
- `results_table.tex` — LaTeX table for the paper

## MVE Success Criteria

| Criterion | Description | Required? |
|-----------|-------------|-----------|
| RL ASR > 0% | Model produces valid attacks | Must pass |
| RL ASR > Random | RL learned something useful | Must pass |
| RL ASR > Template | RL beats fixed templates | Core validation |
| RL ASR > LLM-SingleShot | Iterative RL > single-shot | Core validation |
| RL ASR > Best Baseline + 5% | Substantial improvement | Ideal |

## What's Still Needed

See the section below for items that need to be prepared before running.

### Required Before Training

1. **GPU server access** — A100 40GB minimum. Confirm CUDA 12.1+ and NCCL.
2. **HuggingFace token** — For downloading Qwen2.5-0.5B, Qwen2.5-7B, and Llama-3.1-8B. Llama requires Meta license approval.
3. **wandb account** — For experiment tracking. Set `report_to: "none"` in config to skip.

### Required for Full Experiment (Post-MVE)

4. **MCPTox dataset** — 1,312 test cases from 45 MCP servers. Check if publicly released.
5. **MCP-SafetyBench dataset** — 245 test cases. Check GitHub availability.
6. **GPT-4o API key** — For the LLM judge in the stealth reward (not needed for MVE).
7. **DeBERTa classifier** — Fine-tuned on MCPTox labeled data for the stealth detector ensemble.
8. **Sentence-BERT model** — For diversity reward (SelfBLEU + cosine similarity).
9. **Additional victim models** — Mistral-7B-Instruct-v0.3, GPT-4o-mini API, Claude-3.5-Sonnet API, Gemini-2.0-Flash API.
10. **H100 80GB** — For full experiment with 3B attacker + G=8.
11. **DPO defense training code** — For the closed-loop defense validation (Phase 5).

### Nice to Have

12. **MCP-ITP reproduction** — Their tree-search baseline, if code is released.
13. **MPMA reproduction** — Genetic algorithm baseline with advertising templates.
14. **ToolHijacker reproduction** — Two-phase retrieval-then-selection baseline.
