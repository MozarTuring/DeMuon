export JWM_GPU_NUM=1
export JWM_NODES_NUM=1
if false; then
    #local
    cd /Users/maojingwei/baidu/project/ && source common_tools/meta_script.sh alvis1 DeMuon slurm

rsync -av alvis1:~/project_remote_jwm/fed193bc30219e80a93aec1286e88be4693d184c/ /Users/maojingwei/baidu/project/zzzjwmoutput/Decentralized-Training-Exp/runs/fed193bc30219e80a93aec1286e88be4693d184c
    # remote
    exit
    ssh alvis1
    scancel 6007399
    squeue --me
fi
