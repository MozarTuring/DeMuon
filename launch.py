"""Launcher: reads experiments.toml, spawns one process per experiment,
assigns GPUs round-robin, waits for all to finish."""

import os
import subprocess
import sys
import time
import tomllib


def main():
    config_path = "experiments.toml"
    if not os.path.exists(config_path):
        print(f"ERROR: {config_path} not found. Copy experiments.toml.example and edit it.")
        sys.exit(1)

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    seeds = config["seeds"]
    experiments = config["experiments"]
    n_gpus = int(os.environ.get("JWM_GPU_NUM", "1"))
    n_exps = len(experiments)

    if n_exps > n_gpus:
        print(f"NOTE: {n_exps} experiments on {n_gpus} GPUs — some GPUs will be shared")

    seeds_str = " ".join(str(s) for s in seeds)
    print(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} === "
          f"Launching {n_exps} experiments on {n_gpus} GPUs, seeds={seeds} ===")

    processes = []
    for i, exp in enumerate(experiments):
        name = exp["name"]
        args = exp["args"]
        gpu_id = i % n_gpus
        outdir = f"output/{name}"
        os.makedirs(outdir, exist_ok=True)

        cmd = (f"python exp3_decentralize_gpt.py "
               f"--gpu {gpu_id} --outdir {outdir} --seeds {seeds_str} {args}")

        print(f"[{time.strftime('%H:%M:%S')}] Launching {name} on gpu={gpu_id} -> {outdir}/")
        log_file = open(f"{outdir}/stdout.log", "w")
        proc = subprocess.Popen(
            cmd, shell=True, stdout=log_file, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"})
        processes.append((name, proc, log_file))

    print(f"PIDs: {[p.pid for _, p, _ in processes]}")

    slurm_job_id = os.environ.get("SLURM_JOB_ID", "N/A")
    print(f"Slurm Job ID: {slurm_job_id}")
    print("Monitoring GPU for 5 minutes...")
    try:
        subprocess.run(["nvidia-smi", "-l", "10"], timeout=300)
    except subprocess.TimeoutExpired:
        pass
    except FileNotFoundError:
        print("nvidia-smi not found, skipping GPU monitoring")

    print("=" * 50)
    print(f"Waiting for all {n_exps} jobs to finish...")

    failed = []
    for name, proc, log_file in processes:
        rc = proc.wait()
        log_file.close()
        if rc == 0:
            print(f"[{time.strftime('%H:%M:%S')}] DONE  {name} (pid={proc.pid}) -> SUCCESS")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] DONE  {name} (pid={proc.pid}) -> FAILED (exit {rc})")
            failed.append(name)

    if not failed:
        print(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} === All {n_exps} experiments completed successfully ===")
    else:
        print(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} === {len(failed)} experiments FAILED: {failed} ===")
        print("Check output/*/stdout.log for details")
        sys.exit(1)


if __name__ == "__main__":
    main()
