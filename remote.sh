export JWM_GPU_NUM=1 && export JWM_COMMIT_ID=debug
    scancel --name=debug && sleep 5 && rm -rf ../debug
    if bash check_gpu.sh T4 $2; then
        export JWM_GPU_TYPE=T4
    elif bash check_gpu.sh A40 $2; then
        export JWM_GPU_TYPE=A40
    else
        export JWM_GPU_TYPE=A40
    fi
    echo "$JWM_GPU_TYPE"

    mkdir -p "../${JWM_COMMIT_ID}"
    cp -R . "../${JWM_COMMIT_ID}/"
    cd "../${JWM_COMMIT_ID}"
    sbatch --time=1-00:00:00 --nodes=1 --gpus-per-node=${JWM_GPU_TYPE}:${JWM_GPU_NUM} --job-name="${JWM_COMMIT_ID}" slurm.sh
    while ! [ -f slurm_out.log ]; do sleep 1; done && tail -f slurm_out.log
