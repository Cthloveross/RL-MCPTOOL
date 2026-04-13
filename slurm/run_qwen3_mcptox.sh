#!/bin/bash
#SBATCH --job-name=qwen3-mcptox
#SBATCH --account=xulab
#SBATCH --partition=scavenger-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --exclude=dcc-courses-gpu-[01-10],dcc-carlsonlab-gpu-ferc-s-h36-[23-24],dcc-plusds-gpu-ferc-s-j11-[17-18],dcc-gehmlab-gpu-ferc-s-n32-[10-13,23-24],dcc-gehmlab-gpu-ferc-s-z25-[15,17-19],dcc-pearsonlab-gpu-ferc-s-o15-17,dcc-plusds-gpu-ferc-s-z25-23,dcc-carlsonlab-gpu-ferc-s-o15-10,dcc-chsi-gpu-ferc-s-i11-1
#SBATCH --output=slurm/logs/qwen3_mcptox_%j.out
#SBATCH --error=slurm/logs/qwen3_mcptox_%j.err

echo "Qwen3-8B MCPTox 3-defense run — $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

module load CUDA/12.4
source ~/miniconda3/etc/profile.d/conda.sh
conda activate safemcp

export HF_HOME=/work/tc442/hf_cache
export TRANSFORMERS_CACHE=/work/tc442/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd /work/tc442/RL-MCPTOOL
python -u scripts/mcptox_run_qwen3.py --mode all

echo "Done at $(date)"
