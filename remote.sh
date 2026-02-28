export JWM_GPU_NUM=1
export JWM_NODES_NUM=1
if false; then
    #local
    cd /Users/maojingwei/baidu/project/ && source common_tools/meta_script.sh alvis1 DeMuon slurm

    # remote
    exit
    ssh alvis1
    scancel 6011615
    squeue --me
fi
