#!/usr/bin/bash
#SBATCH -A naiss2026-4-5 -p alvis
#SBATCH --output=slurm_out.log
#SBATCH --error=slurm_out.log

module load Python/3.12.3-GCCcore-13.3.0
source /mimer/NOBACKUP/groups/naiss2025-22-1056/pythonenv/neurips_code/bin/activate

# Experiment args come from remote.sh (already sourced by meta_script.sh
# before sbatch, and copied into RUN_DIR alongside this script).
if [ ! -f remote.sh ]; then
    echo "ERROR: remote.sh not found. Copy remote.sh.example to remote.sh and edit it."
    exit 1
fi
source remote.sh

if [ "${#NETWORKS[@]}" -gt "${JWM_GPU_NUM}" ]; then
    echo "NOTE: ${#NETWORKS[@]} networks on ${JWM_GPU_NUM} GPUs — some GPUs will be shared"
fi

echo "=== $(date) === Starting ${#NETWORKS[@]} parallel runs on ${JWM_GPU_NUM} GPUs ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "============================================="

PIDS=()
for i in "${!NETWORKS[@]}"; do
    net="${NETWORKS[$i]}"
    gpu_id=$((i % JWM_GPU_NUM))
    outdir="output/${net}"
    mkdir -p "${outdir}"

    echo "[$(date)] Launching network=${net} on gpu=${gpu_id} -> ${outdir}/"
    PYTHONUNBUFFERED=1 python exp3_decentralize_gpt.py \
        --network "${net}" \
        --gpu "${gpu_id}" \
        --outdir "${outdir}" \
        ${COMMON_ARGS} \
        > "${outdir}/stdout.log" 2>&1 &
    PIDS+=($!)
done

echo "PIDs: ${PIDS[*]}"
echo "Waiting for all ${#PIDS[@]} jobs to finish..."

FAIL=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    net="${NETWORKS[$i]}"
    if wait "${pid}"; then
        echo "[$(date)] DONE  network=${net} (pid=${pid}) -> SUCCESS"
    else
        rc=$?
        echo "[$(date)] DONE  network=${net} (pid=${pid}) -> FAILED (exit ${rc})"
        FAIL=1
    fi
done

if [ "${FAIL}" -eq 0 ]; then
    echo "=== $(date) === All runs completed successfully ==="
else
    echo "=== $(date) === Some runs FAILED — check output/*/stdout.log ==="
    exit 1
fi
