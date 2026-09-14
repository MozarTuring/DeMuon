# Decentralized GPT Training Experiment

The code implements decentralized training of a small GPT model (MiniGPT) on the Multi30k English–German translation corpus. Workers communicate via gossip-based mixing over configurable graph topologies (ring, exponential, complete) and use the DeMuon optimizer with Newton-Schulz orthogonalization.

## Prerequisites

- Python 3.12+
- CUDA-capable GPU (falls back to CPU if unavailable)

## Installation

```bash
pip install -r requirements_gpt.txt
```

## Reproducing paper results

All 12 experiments (4 algorithms × 3 topologies) are defined in `jwm_configs/experiments.toml`. To run them:

```bash
export JWM_GPU_NUM=4   # number of available GPUs
python launch.py
```

`launch.py` spawns one process per experiment, assigns GPUs round-robin, and waits for all to finish. Per-experiment logs are written to `slurm_out_{name}.log` and results to `output/{name}/`.


