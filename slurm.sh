#!/usr/bin/bash
#SBATCH -A naiss2026-4-5 -p alvis
#SBATCH --output=slurm_out.log
#SBATCH --error=slurm_out.log

module load Python/3.12.3-GCCcore-13.3.0
source /mimer/NOBACKUP/groups/naiss2025-22-1056/pythonenv/neurips_code/bin/activate

if [ ! -f remote.sh ]; then
    echo "ERROR: remote.sh not found. Copy remote.sh.example to remote.sh and edit it."
    exit 1
fi
source remote.sh

python launch.py
