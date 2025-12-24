#!/usr/bin/bash
#SBATCH -A naiss2025-22-1056 -p alvis
#SBATCH -N 1 --gpus-per-node=T4:1
#SBATCH -t 1-00:03:00

module load Python/3.12.3-GCCcore-13.3.0
# python -m venv /mimer/NOBACKUP/groups/naiss2025-22-1056/pythonenv/neurips_code
source /mimer/NOBACKUP/groups/naiss2025-22-1056/pythonenv/neurips_code/bin/activate

cd neurips_code

# pip install --upgrade pip
# pip install -r requirements_gpt.txt
# pip install wandb

python exp3_decentralize_gpt.py
