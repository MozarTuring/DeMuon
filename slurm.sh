#!/usr/bin/bash
#SBATCH -A naiss2026-4-5 -p alvis
#SBATCH --output=slurm_out.log
#SBATCH --error=slurm_out.log

# if [ -d "${jwm_outputdir}" ]; then
#   this dir is passed to sbatch --output and will be created
#   echo "${jwm_outputdir} exists, please chagne"
#   exit
# else
#   mkdir -p "${jwm_outputdir}"
# fi
module load Python/3.12.3-GCCcore-13.3.0
# python -m venv /mimer/NOBACKUP/groups/naiss2025-22-1056/pythonenv/neurips_code
source /mimer/NOBACKUP/groups/naiss2025-22-1056/pythonenv/neurips_code/bin/activate

# pip install --upgrade pip
# pip install -r requirements_gpt.txt
# pip install wandb

PYTHONUNBUFFERED=1 python exp3_decentralize_gpt.py
