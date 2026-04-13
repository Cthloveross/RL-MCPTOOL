#!/bin/bash
#SBATCH --job-name=eval-dpo
#SBATCH --account=xulab
#SBATCH --partition=scavenger-gpu
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/eval_dpo_%j.out
#SBATCH --error=slurm/logs/eval_dpo_%j.err

echo "Eval DPO model — $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

module load CUDA/12.4
source ~/miniconda3/etc/profile.d/conda.sh
conda activate safemcp

export HF_HOME=/work/tc442/hf_cache
export TRANSFORMERS_CACHE=/work/tc442/hf_cache
export TOKENIZERS_PARALLELISM=false

cd /work/tc442/RL-MCPTOOL
python -u scripts/eval_trained_model.py \
    --adapter experiments/mcptox_defender/dpo_checkpoint \
    --tag dpo \
    --modes no_defense

echo "Done at $(date)"
