# MCPDefender

**Beyond Average ASR: Fine-Grained Vulnerability Analysis and RL-Based Defense for MCP Tool Poisoning**

> Tianhao Chen, Tianchen Guan, JiaCheng Sang — April 2026

## Overview

MCPTox (AAAI 2026) reports a 36.5% average attack success rate (ASR) across MCP tool poisoning attacks, but this single number masks three layers of structure: paradigm-level differences (3x gap between explicit hijacking and parameter tampering), tool-category disparities (14pp spread), and model-family divergences (11x gap between most and least vulnerable models).

This project makes two contributions:

1. **Fine-Grained Re-Analysis of MCPTox** — The first paradigm × tool-category × model-family three-way decomposition on MCPTox data (25K+ records, 27 models, 1348 malicious instances). We uncover three findings: *Paradigm Dominance* (P3 Parameter Tampering is 3-5x more effective than P1 Explicit Hijacking), *Inverse Risk Paradox* (communication tools are most vulnerable at 44%, not read-only tools), and *Model-Family Clustering* (vulnerability patterns cluster by family but do not decrease with scale).

2. **MCPDefender** — The first RL-based defense against MCP tool poisoning. We train Qwen3-8B via GRPO (Group Relative Policy Optimization) on seen attack paradigms (T1 Explicit Hijacking + T2 Implicit Hijacking), then evaluate on the held-out paradigm (T3 Parameter Tampering). The core claim: GRPO's online exploration learns generalizable safety reasoning that transfers to unseen attack patterns, outperforming DPO which only memorizes seen-paradigm preferences.

## Project Evolution

| # | Direction | Outcome |
|---|-----------|---------|
| 1 | MCPoisoner (GRPO attack generation) | Abandoned — MCP-ITP/PISmith already done |
| 2 | SafeMCP (multi-turn safety degradation) | Falsified — Position 1 ASR highest (83.3%) |
| 3 | VulnGRPO (low-risk tools more vulnerable) | Falsified — MCPTox shows L3 > L1, Spearman ≈ 0 |
| 4 | **MCPDefender (RL-based defense)** | **Current direction** |

## Project Structure

```
RL-MCPTOOL/
├── configs/
│   ├── mcptox_defender.yaml          # MCPDefender training config
│   ├── mcptox_defense.yaml           # Defense baseline config
│   ├── mcpalign_mve.yaml             # Earlier MVE configs
│   └── vulngrpo_*.yaml               # Earlier VulnGRPO configs
├── data/
│   ├── mcptox_defender/
│   │   ├── sft_data.json             # 2161 SFT examples (T1+T2 only)
│   │   └── dpo_data.json             # 1758 preference pairs
│   ├── mcpalign/                     # Earlier SafeMCP data
│   └── scenarios/                    # Original attack scenarios
├── src/
│   ├── mcpalign/                     # Core library
│   │   ├── llm_judge.py              # GPT-4o-mini LLM-as-judge (81.3% agreement)
│   │   ├── mcptox_data.py            # SFT/DPO data construction from MCPTox
│   │   ├── sft_data.py               # SFT data pipeline
│   │   ├── dpo_data.py               # DPO data pipeline
│   │   ├── environment.py            # MCP-Gym environment
│   │   ├── models.py                 # Model loading (4-bit quantization)
│   │   ├── prompts.py                # Prompt formatting
│   │   ├── reward.py                 # GRPO reward functions
│   │   └── utils.py                  # Config, seeding, logging
│   └── mcpoisoner/                   # Original attack library (deprecated)
├── scripts/
│   ├── mcptox_build_training_data.py # Build SFT/DPO data from MCPTox
│   ├── mcptox_defense_baseline.py    # Zero-shot defense baselines
│   ├── mcptox_run_qwen3.py           # Qwen3-8B inference on MCPTox
│   ├── mcptox_validate_judge.py      # Judge agreement validation
│   ├── mcptox_analyze.py             # Re-analysis statistics
│   ├── eval_trained_model.py         # Evaluate SFT/DPO/GRPO checkpoints
│   ├── mcpalign_train_sft.py         # SFT training
│   ├── mcpalign_train_dpo.py         # DPO training
│   ├── mcpalign_train_grpo.py        # GRPO training
│   └── vulngrpo_*.py                 # Earlier VulnGRPO scripts
├── slurm/                            # SLURM job scripts for DCC
├── experiments/                      # Output (checkpoints, results, figures)
├── docs/                             # Detailed experiment plans & results
│   ├── MCPDefender.md                # Full experiment plan (v4)
│   ├── Exp-Prepare.md                # Execution handbook
│   ├── temp-result.md                # Running experiment log
│   └── dcc-xu-lab.md                 # DCC cluster guide
└── Paper/
    └── proposal.tex                  # Research proposal
```

## Current Status

### Completed

- MCPTox re-analysis: 11,203 mapped records, per-paradigm × per-level ASR computed
- Tool taxonomy: 148 tools, L1-L5 risk levels annotated
- LLM-as-judge: GPT-4o-mini, 81.3% binary agreement with MCPTox ground truth
- Victim model selection: Qwen3-8B Think mode, 4-bit nf4 (no_defense ASR = 22.0%, T3 = 40.3%)
- Zero-shot defense baselines: prompt_hardening (-10pp), defensive_tokens (-8.7pp)
- SFT training: 2161 examples, 3 epochs, loss 1.88 → 0.73, token accuracy 83%
- DPO data: 1758 preference pairs constructed

### In Progress

- DPO training (from SFT checkpoint)
- SFT/DPO evaluation on 150-instance test set

### Upcoming

- GRPO training (~12-18h on A5000 24GB)
- 6-method comparison: No Defense / Prompt Hardening / Defensive Tokens / SFT / DPO / MCPDefender
- Ablation studies (paradigm split, benign ratio, GRPO init, group size)
- Case studies & error analysis

## Key Baseline Results

| Defense | T1 (Explicit) | T2 (Implicit) | T3 (Param Tamper) | ALL |
|---------|:---:|:---:|:---:|:---:|
| No Defense | 22.2% | 0.0% | **40.3%** | 22.0% |
| Prompt Hardening | 11.1% | 1.7% | 20.8% | 12.0% |
| Defensive Tokens | 16.7% | 1.7% | 22.2% | 13.3% |

T3 Parameter Tampering is the hardest to defend (40.3% ASR) and is held out as the unseen test paradigm.

## Setup

### Prerequisites

- **GPU**: NVIDIA A5000 24GB (minimum), A6000 48GB or H200 80GB recommended
- **CUDA**: 12.4+
- **Python**: 3.10+
- **Cluster**: Duke Compute Cluster (DCC), `xulab` account

### Installation

```bash
# 1. Create conda environment
conda create -n safemcp python=3.10 -y
conda activate safemcp

# 2. Install PyTorch (CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Install project
cd /work/tc442/RL-MCPTOOL
pip install -e .

# 4. Pre-download models to /work (avoid home quota)
mkdir -p /work/tc442/hf_cache
export HF_HOME=/work/tc442/hf_cache
```

### SLURM Submission

```bash
# Training
sbatch slurm/run_sft.sh          # SFT training
sbatch slurm/run_dpo.sh          # DPO training

# Evaluation
sbatch slurm/run_eval_sft.sh     # Evaluate SFT checkpoint
sbatch slurm/run_eval_dpo.sh     # Evaluate DPO checkpoint

# Monitoring
squeue -u tc442
tail -f slurm/logs/<job>.out
```

**Note**: Always use `--exclude=dcc-courses-gpu-[01-10]` in SLURM scripts to avoid P100 16GB nodes which lack sufficient VRAM.

## Success Criteria

| Level | Requirements |
|-------|-------------|
| **Top venue (EMNLP/CCS)** | Re-analysis findings significant (p<0.001); MCPDefender unseen-P3 ASR < DPO by ≥10pp; BTSR ≥ 85% |
| **Findings (EMNLP/ACL)** | Re-analysis significant; MCPDefender > DPO on unseen (any margin); 6-method comparison |
| **Workshop (AISec/SaTML)** | Re-analysis alone (C1) or defense comparison alone (C3) |

## Team

| Member | Focus |
|--------|-------|
| Tianhao Chen | ANOVA, SFT, GRPO training, main evaluation, Method + Experiments writing |
| Tianchen Guan | Model clustering, PCA, DPO baseline, ablations, Analysis + Results writing |
| JiaCheng Sang | MCPTox mapping, LLM-as-judge, MCP-Gym, case studies, Introduction + Related Work |
