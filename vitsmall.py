#!/usr/bin/env python3
"""
Train ResNet-34 from scratch on CIFAR-100 with standard CIFAR augmentations.
- Dataset: CIFAR-100
- Model: ResNet-34 (first conv adjusted to 3x3, stride 1; maxpool removed)
- Augmentations: RandomCrop(32, padding=4), RandomHorizontalFlip(), ColorJitter
- Optimizer: SGD(momentum=0.9, weight_decay=5e-4)
- Scheduler: CosineAnnealingLR
- Loss: CrossEntropy with label smoothing (0.1)
- Metrics: Top-1 / Top-5 accuracy
- Mixed precision (AMP) support

Usage:
  python train_resnet34_cifar100.py --data ./data --epochs 200 --batch-size 128 --lr 0.1

"""
from __future__ import annotations
import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet34
from tqdm import tqdm

from utils import *
from timm.models import VisionTransformer, create_model



def vit_small(img_size=32, num_classes=100, drop_path_rate=0.1):
    model = VisionTransformer(img_size=img_size,
                              patch_size=img_size // 8,
                              in_chans=3,
                              num_classes=num_classes,
                              embed_dim=384,
                              depth=12,
                              num_heads=6,
                              drop_path_rate=drop_path_rate)
    return model






# CIFAR-100 channel-wise mean/std (float32, range [0,1])
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True



def get_dataloaders_ordered_splits(
    data_dir: str,
    args,
    workers: int
):
    """
    Return a *list* of train DataLoaders created by splitting the CIFAR-100
    training set into `num_splits` **contiguous, in-order** chunks, plus one
    shared test DataLoader. No randomness is used in the split itself.
    """
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    full_train_set = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    total_len = len(full_train_set)
    base = total_len // args.n_workers
    rem = total_len % args.n_workers

    subsets = []
    start = 0
    for i in range(args.n_workers):
        length = base + (1 if i < rem else 0)
        end = start + length
        idx_range = range(start, end)
        subset = torch.utils.data.Subset(full_train_set, idx_range)
        subsets.append(subset)
        start = end

    train_loaders = [
        DataLoader(
            subset,
            batch_size=args.train_batch_size,
            shuffle=True,           # keep per-node shuffling for SGD
            num_workers=workers,
            pin_memory=True,
            drop_last=True,
        )
        for subset in subsets
    ]

    val_loader = DataLoader(
        test_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=workers, pin_memory=True
    )

    return train_loaders, val_loader


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append((correct_k * (100.0 / batch_size)).item())
        return res


def train_one_epoch(model, loader, optimizer, scaler, device, criterion):
    model.train()
    running_loss = 0.0
    top1_m = 0.0
    top5_m = 0.0
    for images, targets in tqdm(loader, desc='Train', leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        t1, t5 = accuracy(outputs, targets, topk=(1, 5))
        top1_m += t1 * images.size(0)
        top5_m += t5 * images.size(0)

    n = len(loader.dataset)
    return running_loss / n, top1_m / n, top5_m / n


def evaluate(model, loader, device, criterion):
    model.eval()
    running_loss = 0.0
    top1_m = 0.0
    top5_m = 0.0
    with torch.no_grad():
        count = -1
        for images, targets in tqdm(loader, desc='Eval', leave=False):
            count += 1
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            # if count==0:
            #     jwp(images[0,:])
            #     jwp(outputs[0,:])
            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)
            t1, t5 = accuracy(outputs, targets, topk=(1, 5))
            top1_m += t1 * images.size(0)
            top5_m += t5 * images.size(0)

    n = len(loader.dataset)
    return running_loss / n, top1_m / n, top5_m / n


def save_checkpoint(state: dict, is_best: bool, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(state, os.path.join(out_dir, 'last.pth'))
    if is_best:
        torch.save(state, os.path.join(out_dir, 'best.pth'))


    

if __name__ == '__main__':
    jwp("Starting training")
    # parameters defaulted setting
    parser = comoon_args()
    
    # parser = argparse.ArgumentParser(description='ResNet-34 on CIFAR-100 (from scratch)')
    # parser.add_argument('--data', type=str, default='./data', help='dataset root (will download if missing)')
    # parser.add_argument('--epochs', type=int, default=200)
    # parser.add_argument('--batch-size', type=int, default=128)
    # parser.add_argument('--lr', type=float, default=0.1)
    # parser.add_argument('--momentum', type=float, default=0.9)
    # parser.add_argument('--weight-decay', type=float, default=5e-4)
    # parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    # parser.add_argument('--no-amp', action='store_true', help='disable mixed precision')
    # parser.add_argument('--seed', type=int, default=42)
    # parser.add_argument('--out', type=str, default='./checkpoints')
    # parser.add_argument('--resume', type=str, default=None)
    # args = parser.parse_args([])
    
    # args = parser.parse_args(['--n_workers','8','--train_batch_size','128','--eval_batch_size','3500','--network','ring','--alg','muon','--epochs','200'])

    args = parser.parse_args(['--n_workers','1','--train_batch_size','128','--eval_batch_size','3500','--network','','--alg','cenmuon','--epochs','150', '--log_interval','500'])


    wandb.init(project="my-kaggle-project", name=f'{args.alg}_{args.network}_vitsmall')
    artifact = wandb.Artifact("my_model", type="model")
    
    os.makedirs("graphs", exist_ok=True)
    os.makedirs(f"output/{args.network}", exist_ok=True)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loaders, val_loader = get_dataloaders_ordered_splits('./', args, 8)
    iters = [iter(loader) for loader in loaders]


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
    
    workers = []
    base_model = vit_small()
    for _ in range(args.n_workers):
        m = vit_small()
        m.load_state_dict(base_model.state_dict())
        # jwp(m.tok_emb(torch.tensor([0])))
        m.to(device)
        workers.append(m)
    del base_model
    aa=[(ele[0], ele[1].shape) for ele in workers[0].named_parameters()]
    jwp(aa)
    # return "done"
    
    if args.alg == 'dsgd':
        args.lr=1e-2
        lr = args.lr
    elif args.alg == 'dsgd_gclip_decay':
        args.lr=10
        args.l2_clip_bd=0.1
        lr, l2_clip_bd = args.lr, args.l2_clip_bd
    elif args.alg == 'gt_dsgd':
        lr = args.lr
        y_list, g_prev_list = [], []
        for _ in range(args.n_workers):
            # use last m in the memory
            y_list.append({name: torch.zeros_like(param) for name, param in m.named_parameters()})
            g_prev_list.append({name: torch.zeros_like(param) for name, param in m.named_parameters()})
    elif args.alg in ["gt_nsgdm", "muon", "cenmuon"]:
        args.lr=1e-1
        args.mom=0.8
        lr, mom = args.lr, args.mom
        y_list, m_list = [], []
        for _ in range(args.n_workers):
            # use last m in the memory
            y_list.append({name: torch.zeros_like(param) for name, param in m.named_parameters()})
            m_list.append({name: torch.zeros_like(param) for name, param in m.named_parameters()})
    elif args.alg == 'sen':
        lr, mom, phi, tau = args.lr, args.mom, args.phi, args.tau
        m_list = []
        for _ in range(args.n_workers):
            m_list.append({name: torch.zeros_like(param) for name, param in m.named_parameters()})
    # if args.alg == 'adamw':
    #     # what's the warmup strategy for adamw?
    #     optimizer = optim.AdamW(workers[0].parameters(), lr=1e-3, weight_decay=0.05, betas=(0.9, 0.999), eps=1e-8)
        
    total_params = sum(p.numel() for p in workers[0].parameters())
    jwp(f"total_params = {total_params}")
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # 6) Example mixing matrix: uniform average
    # W = torch.full((args.n_workers, args.n_workers), 1/args.n_workers, device="cuda")

    # 7) CSV logging
    header = ["round"]
    header += [f"w{i}_val"   for i in range(args.n_workers)]
    header += [f"w{i}_train" for i in range(args.n_workers)]
    loss_table = [header]

    max_round_per_epoch = len(loaders[0])
    total_rounds = args.epochs * max_round_per_epoch
    for r in range(1, total_rounds+1):
        round_losses = []
        # one local step per worker
        for wid, model in enumerate(workers):
            try:
                x, y = next(iters[wid])
            except StopIteration:          # start a new local epoch
                iters[wid] = iter(loaders[wid]) # reset the loader
                x, y = next(iters[wid])

            x, y = x.to(device), y.to(device)

            # ----- forward/backward -----
            model.train()
            logits = model(x)
            # jwp(f"logits shape: {logits.shape}, y shape: {y.shape}")
            loss   = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            model.zero_grad(set_to_none=True)
            loss.backward()

            # use local gradient to update local buffers
            if args.alg == 'dsgd':
                with torch.no_grad():
                    # update buffers block by block
                    for (name, p) in model.named_parameters():
                        if p.grad is None:
                            continue
                        g = p.grad
                        p.data -= lr * g
            elif args.alg == 'dsgd_gclip_decay':
                # clip gradient norm globally
                lr = args.lr / r
                l2_clip_bd = args.l2_clip_bd * r**0.4
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=l2_clip_bd)
                # update buffers block by block
                with torch.no_grad():
                    for (name, p) in model.named_parameters():
                        if p.grad is None:
                            continue
                        g = p.grad
                        p.data -= lr * g 
            elif args.alg == 'gt_dsgd':
                with torch.no_grad():
                    # update buffers block by block
                    for (name, p) in model.named_parameters():
                        if p.grad is None:
                            continue
                        g = p.grad
                        g_prev = g_prev_list[wid][name]
                        y = y_list[wid][name]
                        # buffer update
                        y.add_(g).add_(g_prev, alpha=-1.0)
                        y_list[wid][name] = y
                        g_prev_list[wid][name] = g
            elif args.alg in ["gt_nsgdm","muon",'cenmuon']: 
                with torch.no_grad():
                    # update buffers block by block
                    for (name, p) in model.named_parameters():
                        if p.grad is None:
                            continue
                        g = p.grad
                        m = m_list[wid][name]
                        y = y_list[wid][name]
                        # buffer update
                        m_temp = m.mul(mom).add(g, alpha=1-mom) # get v^t
                        y.add_(m_temp).add_(m, alpha=-1.0)
                        y_list[wid][name] = y
                        m_list[wid][name] = m_temp 
            elif args.alg == 'sen':
                with torch.no_grad():
                    # update buffers block by block
                    for (name, p) in model.named_parameters():
                        if p.grad is None:
                            continue
                        g = p.grad
                        m = m_list[wid][name]
                        # buffer update
                        temp = sclip(g.add(m, alpha=-1.0), phi, r, tau)
                        m.mul_(mom/r**0.5).add_(temp, alpha=1-mom/r**0.5)
                        m_list[wid][name] = m
                        p.data -= lr/r**0.2 * m
            round_losses.append(loss.item())

        # ---- mixing after every local step, i.e., communication after every local step  -----
        if args.alg == 'dsgd' or args.alg == 'dsgd_gclip_decay':
            mix_params(workers, mixing)
        elif args.alg == "gt_dsgd":
            y_list = mix_y_list(y_list, mixing)
            # use y_list to update parameters on each worker
            for wid, model in enumerate(workers):
                with torch.no_grad():
                    # update parameters block by block
                    for (name, p) in model.named_parameters():
                        if p.grad is None:
                            continue
                        p.data -= lr * y_list[wid][name]
            mix_params(workers, mixing)
        elif args.alg in ["gt_nsgdm", "muon"]:
            # mix y_list
            y_list = mix_y_list(y_list, mixing)
            # use normalized y_list to update parameters on each worker
            svd_time=0.0
            for wid, model in enumerate(workers):
                # normalize model's y globally
                if args.alg == "gt_nsgdm":
                    normalized_y = normalize_tensor_dict(y_list[wid])
                elif args.alg == "muon":
                    normalized_y = dict()
                    for name in y_list[wid]:
                        tmp=y_list[wid][name].squeeze()
                        if tmp.ndim==1 or name in ["tok_emb","pos_emg", "patch_embed.proj.weight"]:
                            normalized_y[name] = tmp/torch.norm(tmp)
                        elif tmp.ndim==2:
                            tmp_start = time.time()
                            # U, S, Vt = torch.linalg.svd(tmp, full_matrices=False)
                            # svd_time+=time.time()-tmp_start
                            # normalized_y[name] = U @ Vt
                            normalized_y[name]=newtonschulz5(tmp)
                        else:
                            jwp(f"Error: not implemented for {name} {tmp.shape} ndim>2")
                            aaa=1/0
                with torch.no_grad():
                    # update parameters block by block
                    for (name, p) in model.named_parameters():
                        if p.grad is None:
                            continue
                        # if name == "pos_emb":
                        #     jwp(p.data.shape,normalized_y[name].shape)
                        #     return None
                        p.data -= lr * normalized_y[name]
            mix_params(workers, mixing)
        elif args.alg == 'sen':
            mix_params(workers, mixing)

        elif args.alg == 'cenmuon':
            normalized_y = dict()
            for name in y_list[wid]:
                tmp=y_list[wid][name].squeeze()
                if tmp.ndim==1 or name in ["tok_emb","pos_emg", "patch_embed.proj.weight"]:
                    normalized_y[name] = tmp/torch.norm(tmp)
                elif tmp.ndim==2:
                    tmp_start = time.time()
                    # U, S, Vt = torch.linalg.svd(tmp, full_matrices=False)
                    # svd_time+=time.time()-tmp_start
                    # normalized_y[name] = U @ Vt
                    normalized_y[name]=newtonschulz5(tmp)
                else:
                    jwp(f"Error: not implemented for {name} {tmp.shape} ndim>2")
                    aaa=1/0
            with torch.no_grad():
                # update parameters block by block
                for (name, p) in model.named_parameters():
                    if p.grad is None:
                        continue
                    # if name == "pos_emb":
                    #     jwp(p.data.shape,normalized_y[name].shape)
                    #     return None
                    p.data -= lr * normalized_y[name]

       
        # logging
        # jwp(f"svd_time={svd_time:.4f}")
        if r % args.log_interval == 0 or r == total_rounds or r == 1:
            val_losses = [evaluate(m, val_loader, device, loss_fn) for m in workers]
            loss_table.append([r] + val_losses + round_losses)
            jwp(f"Round {r}/{total_rounds}: {round_losses},{val_losses}")
            
    # after the for‑round loop
    end(args, artifact, loss_table)
