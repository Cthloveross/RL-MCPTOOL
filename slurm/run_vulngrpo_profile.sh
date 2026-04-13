#!/bin/bash
#SBATCH --job-name=vulngrpo
#SBATCH --account=xulab
#SBATCH --partition=scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --exclude=dcc-courses-gpu-[01-10]
#SBATCH --output=slurm/logs/vulngrpo_%j.out
#SBATCH --error=slurm/logs/vulngrpo_%j.err

echo "=========================================="
echo "VulnGRPO Profiling — $(date)"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Model: ${MODEL:-all}"
echo "=========================================="

module load CUDA/12.4

source ~/miniconda3/etc/profile.d/conda.sh
conda activate safemcp

export HF_HOME=/work/tc442/hf_cache
export TRANSFORMERS_CACHE=/work/tc442/hf_cache
export TOKENIZERS_PARALLELISM=false

cd /work/tc442/RL-MCPTOOL

if [ -n "$MODEL" ]; then
    python scripts/vulngrpo_profile.py --config configs/vulngrpo_profile.yaml --model "$MODEL"
else
    python scripts/vulngrpo_profile.py --config configs/vulngrpo_profile.yaml
fi

EXIT_CODE=$?
echo ""
echo "=========================================="
echo "Job finished at $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="
exit $EXIT_CODE
