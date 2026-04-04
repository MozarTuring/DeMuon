# Decentralized GPT Training Experiment

This repository implements decentralized training of a small GPT model (MiniGPT) on the Multi30k English–German translation corpus. Workers communicate via gossip-based mixing over configurable graph topologies (ring, exponential, complete) and use the DeMuon optimizer with Newton-Schulz orthogonalization.

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

The exact hyperparameters per method and topology are:

| Experiment | Algorithm | Network | lr | lr_schedule | mom |
|---|---|---|---|---|---|
| `dsgd_dim2_complete` | dsgd | complete | 0.006 | 2 (`η/√k`) | — |
| `dsgd_clip_dim2_complete` | dsgd_gclip_decay | complete | 0.6 | 4 (`η/k`) | — |
| `gt_nsgdm2_complete` | gt_nsgdm | complete | 0.07 | 3 (constant) | 0.8 |
| `demuon_decay_complete` | demuon | complete | 0.1 | 2 (`η/√k`) | 0.2 |
| `dsgd_lin1_exp` | dsgd | exp | 0.03 | 1 (linear) | — |
| `dsgd_clip_lin2_exp` | dsgd_gclip_decay | exp | 0.2 | 1 (linear) | — |
| `gt_nsgdm_lin2_exp` | gt_nsgdm | exp | 0.05 | 1 (linear) | 0.8 |
| `demuon_lin1_exp` | demuon | exp | 0.005 | 1 (linear) | 0.8 |
| `dsgd_lin2_ring` | dsgd | ring | 0.03 | 1 (linear) | — |
| `dsgd_clip_lin6_ring` | dsgd_gclip_decay | ring | 0.1 | 1 (linear) | — |
| `gt_nsgdm_lin2_ring` | gt_nsgdm | ring | 0.03 | 1 (linear) | 0.8 |
| `demuon_lin1_ring` | demuon | ring | 0.003 | 1 (linear) | 0.8 |

All runs use seed 42 and `--msgn 1` (Newton-Schulz) for DeMuon, `--l2_clip_bd 0.1` for DSGD_Clip.

To generate figures from the results:

```bash
python plot_results.py --datadir output/
```
