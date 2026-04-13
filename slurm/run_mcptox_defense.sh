#!/bin/bash
#SBATCH --job-name=mcptox-def
#SBATCH --account=xulab
#SBATCH --partition=scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --exclude=dcc-courses-gpu-[01-10],dcc-carlsonlab-gpu-ferc-s-h36-[23-24],dcc-plusds-gpu-ferc-s-j11-[17-18],dcc-gehmlab-gpu-ferc-s-n32-[10-13,23-24],dcc-gehmlab-gpu-ferc-s-z25-[15,17-19],dcc-pearsonlab-gpu-ferc-s-o15-17,dcc-plusds-gpu-ferc-s-z25-23,dcc-carlsonlab-gpu-ferc-s-o15-10,dcc-chsi-gpu-ferc-s-i11-1
#SBATCH --output=slurm/logs/mcptox_def_%j.out
#SBATCH --error=slurm/logs/mcptox_def_%j.err

echo "=========================================="
echo "MCPTox Defense Baseline — $(date)"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=========================================="

module load CUDA/12.4

source ~/miniconda3/etc/profile.d/conda.sh
conda activate safemcp

export HF_HOME=/work/tc442/hf_cache
export TRANSFORMERS_CACHE=/work/tc442/hf_cache
export TOKENIZERS_PARALLELISM=false

cd /work/tc442/RL-MCPTOOL
python scripts/mcptox_defense_baseline.py --config configs/mcptox_defense.yaml --max-per-cell ${MAX_PER_CELL:-2}

EXIT_CODE=$?
echo ""
echo "=========================================="
echo "Job finished at $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="
exit $EXIT_CODE
