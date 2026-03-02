export JWM_GPU_NUM=2
export JWM_NODES_NUM=1
export JWM_RUN_TIME="1-00:00:00"

# Optional: meaningful tag to distinguish runs under the same commit.
# If unset, meta_script.sh auto-generates one from the timestamp.
# export JWM_RUN_TAG="ring_seeds3"
# export JWM_RUN_TAG="ablation_no_msgn"
# export JWM_RUN_TAG="final_3topo"

# Experiment args (used by slurm.sh)
NETWORKS=(complete exp ring)
SEEDS="42"
COMMON_ARGS="--alg demuon --msgn 1 --seeds ${SEEDS}"

if false; then
    #local
    cd /Users/maojingwei/baidu/project/ && source common_tools/meta_script.sh alvis1 DeMuon slurm

    # remote
    exit
    ssh alvis1
    scancel 6022218
    squeue --me
fi

if false; then
    rsync -av alvis1:~/project_remote_jwm/runs/DeMuon/75e38fab2adf1e39629f12360a5ff12eb4ad4eb7/ /Users/maojingwei/baidu/project/zzzjwmoutput/DeMuon/runs/75e38fab2adf1e39629f12360a5ff12eb4ad4eb7/
fi

if false; then
    rsync -av alvis1:~/project_remote_jwm/runs/DeMuon/c6e197aa99933f681dd64b87c7a3fee5de3dd464/ /Users/maojingwei/baidu/project/zzzjwmoutput/DeMuon/runs/c6e197aa99933f681dd64b87c7a3fee5de3dd464/
fi

