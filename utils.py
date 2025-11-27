import torch
from copy import deepcopy
from typing import Union
import wandb
from tqdm import tqdm
import math, random, argparse, csv
from pathlib import Path

import numpy as np
from graph import connected_cycle_weights, exponential_graph_weights, complete_graph_weights

import os
import time

import sys, logging
import torch.nn as nn

# 1) Remove any existing handlers (Jupyter often adds one)
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)

# 2) Build a fresh console handler to stdout
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - %(pathname)s - LINE%(lineno)d - \n%(message)sMSG-END', '%Y-%m-%d %H:%M:%S'))

# 3) Attach to root and set levels
root = logging.getLogger()
root.setLevel(logging.INFO)
root.addHandler(handler)
def jwp(*args):
    total = ''
    for ele in args:
        total.join(f'{ele}, ')
    logging.info(total)
jwp = logging.info

wandb.login(key='44605eadd683b5ce6dd038f6678b22fbbe392f3f')

# ---------- mixing utility ------------------------------------
def mix_params(workers, mixing):
    with torch.no_grad():
        # snapshot all params first
        states = [dict(w.state_dict()) for w in workers]

        for i, w in enumerate(workers):
            new_state = {}
            for k, p0 in w.state_dict().items():
                stacked = torch.stack([s[k].float() for s in states])  # [P, …]

                # add enough singleton dims for broadcasting
                weight = mixing[i].view(-1, *([1] * (stacked.dim() - 1)))

                new_state[k] = torch.sum(weight * stacked, dim=0).type_as(p0)
            w.load_state_dict(new_state)

def normalize_tensor_dict(tensor_dict):
    """
    Normalize a dictionary of tensors so that the global L2 norm is 1.0.
    
    Args:
        tensor_dict (dict): A dictionary {name: tensor}
        
    Returns:
        dict: A new dictionary with same keys, values normalized
    """
    normed_dict = deepcopy(tensor_dict)

    # Compute global L2 norm
    norm = torch.sqrt(sum((v ** 2).sum() for v in normed_dict.values()))
    
    # Normalize if norm is non-zero
    if norm > 0:
        for name in normed_dict:
            normed_dict[name] /= norm
    
    return normed_dict

def mix_y_list(y_list, mixing):
    """
    Mix a list of dictionaries of tensors.
    
    Args:
        y_list (list): A list of dictionaries {name: tensor}
        mixing (torch.Tensor): A mixing matrix of shape (n_workers, n_workers)
        
    Returns:
        list: A new list of dictionaries with same keys, values mixed
    """ 
    mixed_y_list = []
    for i, y in enumerate(y_list):
        mixed_y = {}
        for name in y: # mixing block by block
            stacked = torch.stack([s[name].float() for s in y_list])
            weight = mixing[i].view(-1, *([1] * (stacked.dim() - 1)))
            mixed_y[name] = torch.sum(weight * stacked, dim=0).type_as(y[name])
        mixed_y_list.append(mixed_y)
    return mixed_y_list

def sclip(
    y: torch.Tensor,
    phi: Union[float, torch.Tensor],
    t:   Union[float, torch.Tensor],
    tau: Union[float, torch.Tensor],
) -> torch.Tensor:
    """
    Apply  f(y) = y*phi / sqrt( y^2 * (t+1) + tau * (t+1)^1.6 )
    component‑wise to a tensor y.

    Args
    ----
    y   : torch.Tensor          (any shape)
    phi : float or 0‑D tensor
    t   : float or 0‑D tensor
    tau : float or 0‑D tensor

    Returns
    -------
    torch.Tensor  (same shape as y)
    """
    # make sure parameters are tensors on the same device / dtype as y
    phi = torch.as_tensor(phi, dtype=y.dtype, device=y.device)
    t   = torch.as_tensor(t,   dtype=y.dtype, device=y.device)
    tau = torch.as_tensor(tau, dtype=y.dtype, device=y.device)

    denom = torch.sqrt( y**2 * (t + 1.0) + tau * (t + 1.0)**1.6 )
    return y * phi / denom




def newtonschulz5(G, steps=5, eps=1e-7):
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X



def comoon_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_workers", type=int, required=True)
    parser.add_argument("--epochs",    type=int, required=True)
    parser.add_argument("--train_batch_size",type=int, required=True)
    parser.add_argument("--eval_batch_size",type=int, required=True)
    parser.add_argument("--log_interval",type=int, required=True)
    parser.add_argument("--random_seed", type=int, default=42)
    
    parser.add_argument("--network", type=str, required=True)
    parser.add_argument("--alg",       type=str, required=True)

    return parser



def end(args, artifact, loss_table):
    if args.alg == 'dsgd':
        out_csv = Path(f"output/{args.network}/{args.network}_dsgd_lr{lr}_epoch{args.epochs}_seed{args.random_seed}_worker_losses.csv")
    elif args.alg == 'dsgd_gclip_decay':
        out_csv = Path(f"output/{args.network}/{args.network}_dsgd_gclip_decay_lr{args.lr}_l2_clip_bd{args.l2_clip_bd}_epoch{args.epochs}_seed{args.random_seed}_worker_losses.csv")
    elif args.alg == "gt_dsgd":
        out_csv = Path(f"output/{args.network}/{args.network}_gt_dsgd_lr{lr}_epoch{args.epochs}_seed{args.random_seed}_worker_losses.csv")
    elif args.alg in ["gt_nsgdm", "muon", "cenmuon"]:
        out_csv = Path(f"output/{args.network}/{args.network}_{args.alg}_lr{args.lr}_mom{args.mom}_epoch{args.epochs}_seed{args.random_seed}_worker_losses.csv")
    elif args.alg == 'sen':
        out_csv = Path(f"output/{args.network}/{args.network}_sen_lr{lr}_mom{mom}_phi{args.phi}_tau{args.tau}_epoch{args.epochs}_seed{args.random_seed}_worker_losses.csv")
    else:
        out_csv = Path(f"output/{args.network}_{args.alg}_losses.csv")
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(loss_table)

    artifact.add_file(str(out_csv))
    wandb.log_artifact(artifact)

    jwp(f"Done. Losses saved to {out_csv}")



def evaluate(model, loader, device, criterion):
    model.eval()
    running_loss = 0.0
    top1_m = 0.0
    top5_m = 0.0
    with torch.no_grad():
        count = -1
        n = 0
        for images, targets in tqdm(loader, desc='Eval', leave=False):
            count += 1
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            # if count==0:
            #     jwp(images[0,:])
            #     jwp(outputs[0,:])
            n += images.size(0)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * images.size(0)
            if isinstance(criterion, nn.MSELoss):
                t1, t5 = 0, 0
            else:
                t1, t5 = accuracy(outputs, targets, topk=(1, 5))
                top1_m += t1 * images.size(0)
                top5_m += t5 * images.size(0)

    # n = len(loader.dataset)
    return running_loss / n, top1_m / n, top5_m / n




def get_graph(args, device):
    if args.network == "ring":
        weights = connected_cycle_weights(filename=f"graphs/ring_{args.n_workers}.npy", n=args.n_workers, degree=1)
        mixing = torch.from_numpy(weights).float().to(device)
        # ring: 0.804737854124365
    elif args.network == "exp":
        weights = exponential_graph_weights(filename=f"graphs/exp_{args.n_workers}.npy", n=args.n_workers)
        mixing = torch.from_numpy(weights).float().to(device)
        # exp:  0.5999999999999998 
    elif args.network == "complete":
        weights = complete_graph_weights(filename=f"graphs/complete_{args.n_workers}.npy", n=args.n_workers)
        mixing = torch.from_numpy(weights).float().to(device)
        # complete:

    return mixing, weights



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
