#!/bin/bash
#SBATCH --job-name=bf16-base
#SBATCH --account=xulab
#SBATCH --partition=scavenger-gpu
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/bf16_baseline_%j.out
#SBATCH --error=slurm/logs/bf16_baseline_%j.err

echo "Qwen3-8B bf16 baseline — $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

module load CUDA/12.4
source ~/miniconda3/etc/profile.d/conda.sh
conda activate safemcp

export HF_HOME=/work/tc442/hf_cache
export TRANSFORMERS_CACHE=/work/tc442/hf_cache
export TOKENIZERS_PARALLELISM=false

cd /work/tc442/RL-MCPTOOL

# 150 sample, 3 defense modes, bf16
python -u scripts/eval_bf16_baseline.py --modes no_defense prompt_hardening defensive_tokens

echo "Done at $(date)"
