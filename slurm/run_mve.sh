#!/bin/bash
#SBATCH --job-name=safemcp-mve
#SBATCH --account=xulab
#SBATCH --partition=scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --exclude=dcc-courses-gpu-[01-10]
#SBATCH --output=slurm/logs/mve_%j.out
#SBATCH --error=slurm/logs/mve_%j.err

echo "=========================================="
echo "SafeMCP MVE — $(date)"
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=========================================="

# ── Modules ──
module load CUDA/12.4

# ── Environment ──
source ~/miniconda3/etc/profile.d/conda.sh
conda activate safemcp

# ── HuggingFace cache ──
export HF_HOME=/work/tc442/hf_cache
export TRANSFORMERS_CACHE=/work/tc442/hf_cache
export TOKENIZERS_PARALLELISM=false

# ── Run MVE ──
cd /work/tc442/RL-MCPTOOL
python scripts/mcpalign_mve.py --config configs/mcpalign_mve.yaml

EXIT_CODE=$?
echo ""
echo "=========================================="
echo "Job finished at $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

# Print results if they exist
if [ -f experiments/mcpalign_mve/mve_results.json ]; then
    echo ""
    echo "=== Results JSON ==="
    cat experiments/mcpalign_mve/mve_results.json
fi

exit $EXIT_CODE
