#!/bin/bash
#SBATCH --job-name=mcpdef-grpo
#SBATCH --account=xulab
#SBATCH --partition=scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --output=slurm/logs/grpo_%j.out
#SBATCH --error=slurm/logs/grpo_%j.err
#SBATCH --exclude=dcc-courses-gpu-[01-10]

echo "MCPDefender GRPO Training — $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

module load CUDA/12.4
source ~/miniconda3/etc/profile.d/conda.sh
conda activate safemcp

export HF_HOME=/work/tc442/hf_cache
export TRANSFORMERS_CACHE=/work/tc442/hf_cache
export TOKENIZERS_PARALLELISM=false

cd /work/tc442/RL-MCPTOOL

# Full run: Qwen3-8B from SFT checkpoint
python -u scripts/mcptox_train_grpo.py \
    --config configs/mcptox_defender.yaml \
    --sft-checkpoint experiments/mcptox_defender/sft_checkpoint

echo "Done at $(date)"
