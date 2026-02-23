scancel --name=debug && sleep 5 && rm -rf ../debug
export JWM_GPU_TYPE=A40
if bash check_gpu.sh T4 $JWM_GPU_NUM; then
    export JWM_GPU_TYPE=T4
fi
if bash check_gpu.sh A40 $JWM_GPU_NUM; then
    export JWM_GPU_TYPE=A40
fi
echo $JWM_GPU_TYPE

export JWM_COMMIT_ID=${JWM_COMMIT_ID_TMP} && [ ! -d ../${JWM_COMMIT_ID} ] && mkdir -p ../${JWM_COMMIT_ID} && cp -R . ../${JWM_COMMIT_ID}/ && cd ../${JWM_COMMIT_ID}
sbatch --time=1-00:00:00  --nodes=1 --gpus-per-node=${JWM_GPU_TYPE}:${JWM_GPU_NUM} --job-name=${JWM_COMMIT_ID} slurm.sh ${JWM_GPU_NUM}
while ! [ -f slurm_out.log ]; do sleep 1; done; tail -f slurm_out.log
