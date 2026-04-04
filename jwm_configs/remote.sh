export JWM_GPU_NUM=4
export JWM_NODES_NUM=1
export JWM_RUN_TIME="1-00:00:00"
export JWM_GPU_TYPE=A40

# Optional: meaningful tag to distinguish runs under the same commit.
# If unset, meta_script.sh auto-generates a timestamp.
# export JWM_RUN_TAG="final_all_exps"

if false; then
    #local
    cd /Users/maojingwei/baidu/project/ && source common_tools/meta_script.sh alvis1 DeMuon remote_slurm

    # remote
    exit
    ssh alvis1
    scancel 6083873; squeue --me
fi


