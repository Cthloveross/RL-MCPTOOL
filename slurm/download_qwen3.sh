#!/bin/bash
#SBATCH --job-name=dl-qwen3
#SBATCH --account=xulab
#SBATCH --partition=common
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/dl_qwen3_%j.out
#SBATCH --error=slurm/logs/dl_qwen3_%j.err

echo "Starting Qwen3-8B download at $(date) on $(hostname)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate safemcp

export HF_HOME=/work/tc442/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=0

python -u -c "
from dotenv import load_dotenv
load_dotenv('/work/tc442/RL-MCPTOOL/.env')
from huggingface_hub import snapshot_download
import time
t = time.time()
path = snapshot_download('Qwen/Qwen3-8B', max_workers=2)
print(f'Done in {time.time()-t:.0f}s:', path)
"

echo "Finished at $(date)"
