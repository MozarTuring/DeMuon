# Decentralized GPT Training Experiment

This repository implements decentralized training of a small GPT model (MiniGPT) on the Multi30k English–German translation corpus. Workers communicate via gossip-based mixing over configurable graph topologies (ring, exponential, complete) and use the DeMuon optimizer with Newton-Schulz orthogonalization.

## Prerequisites

- Python 3.12+
- CUDA-capable GPU (falls back to CPU if unavailable)
- A [Weights & Biases](https://wandb.ai) account (logging is enabled by default)

## Installation

```bash
pip install -r requirements_gpt.txt
pip install wandb
```

## Running `exp3_decentralize_gpt.py`

### Environment variable

The script requires a `jwm_commit_id` environment variable, used to name the W&B run. Set it before launching:

```bash
export jwm_commit_id="my_run_v1"
```

> If the value contains the substring `"test"`, training will stop early (after round 10) for quick sanity checks.

### Basic usage

```bash
python exp3_decentralize_gpt.py
```

This runs with all default hyperparameters (8 workers, ring topology, 12 epochs, etc.).

### Full argument reference

| Argument | Type | Default | Description |
|---|---|---|---|
| `--block_size` | int | 64 | Context window length for each training sample |
| `--d_model` | int | 256 | Transformer hidden dimension |
| `--n_layer` | int | 2 | Number of transformer decoder layers |
| `--n_head` | int | 4 | Number of attention heads |
| `--max_len` | int | 128 | Maximum positional embedding length |
| `--train_batch_size` | int | 64 | Per-worker training batch size |
| `--eval_batch_size` | int | 512 | Evaluation batch size |
| `--epochs` | int | 12 | Number of training epochs |
| `--log_interval` | int | 100 | Evaluate and log every N rounds |
| `--n_workers` | int | 8 | Number of decentralized workers |
| `--lr` | float | 0.1 | Learning rate |
| `--mom` | float | 0.8 | Momentum coefficient |
| `--msgn` | int | 1 | Update normalization: 0 = plain, 1 = Newton-Schulz, 2 = SVD |
| `--lr_schedule` | int | 3 | LR schedule: 1 = linear decay, 2 = 1/sqrt(t), 3 = constant |
| `--network` | str | `ring` | Graph topology (`ring`, `exp`, `complete`) |
| `--alg` | str | `DeMuon` | Algorithm name (used for W&B tagging) |

### Example

```bash
export jwm_commit_id="ring_8w_lr01"

python exp3_decentralize_gpt.py \
  --n_workers 8 \
  --network ring \
  --lr 0.1 \
  --mom 0.8 \
  --epochs 12 \
  --train_batch_size 64 \
  --msgn 1 \
  --lr_schedule 3
```


