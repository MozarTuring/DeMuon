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
from math import ceil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet34
from tqdm import tqdm
import torch.nn.functional as F

from utils import *
from timm.models import VisionTransformer, create_model





def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)


def batch_crop(images, crop_size):
    r = (images.size(-1) - crop_size)//2
    shifts = torch.randint(-r, r+1, size=(len(images), 2), device=images.device)
    images_out = torch.empty((len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype)
    # The two cropping methods in this if-else produce equivalent results, but the second is faster for r > 2.
    if r <= 2:
        for sy in range(-r, r+1):
            for sx in range(-r, r+1):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                images_out[mask] = images[mask, :, r+sy:r+sy+crop_size, r+sx:r+sx+crop_size]
    else:
        images_tmp = torch.empty((len(images), 3, crop_size, crop_size+2*r), device=images.device, dtype=images.dtype)
        for s in range(-r, r+1):
            mask = (shifts[:, 0] == s)
            images_tmp[mask] = images[mask, :, r+s:r+s+crop_size, :]
        for s in range(-r, r+1):
            mask = (shifts[:, 1] == s)
            images_out[mask] = images_tmp[mask, :, :, r+s:r+s+crop_size]
    return images_out

class CifarLoader:

    def __init__(self, path, train=True, batch_size=500, aug=None):
        CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
        CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616))
        data_path = os.path.join(path, 'train.pt' if train else 'test.pt')
        if not os.path.exists(data_path):
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            torch.save({'images': images, 'labels': labels, 'classes': dset.classes}, data_path)

        data = torch.load(data_path, map_location=torch.device('cuda'))
        self.images, self.labels, self.classes = data['images'], data['labels'], data['classes']
        # It's faster to load+process uint8 data than to load preprocessed fp16 data
        self.images = (self.images.half() / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)
        # self.images = (self.images / 255).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)

        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
        self.proc_images = {} # Saved results of image processing to be done on the first epoch
        self.epoch = 0

        self.aug = aug or {}
        for k in self.aug.keys():
            assert k in ['flip', 'translate'], 'Unrecognized key: %s' % k

        self.batch_size = batch_size
        self.drop_last = train
        self.shuffle = train

    def __len__(self):
        return len(self.images)//self.batch_size if self.drop_last else ceil(len(self.images)/self.batch_size)

    def __iter__(self):

        if self.epoch == 0:
            images = self.proc_images['norm'] = self.normalize(self.images)
            # Pre-flip images in order to do every-other epoch flipping scheme
            if self.aug.get('flip', False):
                images = self.proc_images['flip'] = batch_flip_lr(images)
            # Pre-pad images to save time when doing random translation
            pad = self.aug.get('translate', 0)
            if pad > 0:
                self.proc_images['pad'] = F.pad(images, (pad,)*4, 'reflect')

        if self.aug.get('translate', 0) > 0:
            images = batch_crop(self.proc_images['pad'], self.images.shape[-2])
        elif self.aug.get('flip', False):
            images = self.proc_images['flip']
        else:
            images = self.proc_images['norm']
        # Flip all images together every other epoch. This increases diversity relative to random flipping
        if self.aug.get('flip', False):
            if self.epoch % 2 == 1:
                images = images.flip(-1)

        self.epoch += 1

        indices = (torch.randperm if self.shuffle else torch.arange)(len(images), device=images.device)
        for i in range(len(self)):
            idxs = indices[i*self.batch_size:(i+1)*self.batch_size]
            yield (images[idxs], self.labels[idxs])

def zeropower_via_newtonschulz5(G, steps=3, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.6, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=1-momentum)
        self.weight.requires_grad = False
        # Note that PyTorch already initializes the weights to one and bias to zero

class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=3, padding='same', bias=False)

    def reset_parameters(self):
        super().reset_parameters()
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])

class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = Conv(channels_in,  channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x


class CifarNet(nn.Module):
    def __init__(self):
        super().__init__()
        widths = dict(block1=64, block2=256, block3=256)
        whiten_kernel_size = 2
        whiten_width = 2 * 3 * whiten_kernel_size**2
        self.whiten = nn.Conv2d(3, whiten_width, whiten_kernel_size, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(
            nn.GELU(),
            ConvGroup(whiten_width,     widths['block1']),
            ConvGroup(widths['block1'], widths['block2']),
            ConvGroup(widths['block2'], widths['block3']),
            nn.MaxPool2d(3),
        )
        self.head = nn.Linear(widths['block3'], 10, bias=False)
        for mod in self.modules():
            if isinstance(mod, BatchNorm):
                mod.float()
            else:
                mod.half()

    def reset(self):
        for m in self.modules():
            if type(m) in (nn.Conv2d, Conv, BatchNorm, nn.Linear):
                m.reset_parameters()
        w = self.head.weight.data
        w *= 1 / w.std()

    def init_whiten(self, train_images, eps=5e-4):
        c, (h, w) = train_images.shape[1], self.whiten.weight.shape[2:]
        patches = train_images.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1,c,h,w).float()
        patches_flat = patches.view(len(patches), -1)
        est_patch_covariance = (patches_flat.T @ patches_flat) / len(patches_flat)
        eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO='U')
        eigenvectors_scaled = eigenvectors.T.reshape(-1,c,h,w) / torch.sqrt(eigenvalues.view(-1,1,1,1) + eps)
        self.whiten.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))

    def forward(self, x, whiten_bias_grad=True):
        b = self.whiten.bias
        x = F.conv2d(x, self.whiten.weight, b if whiten_bias_grad else b.detach())
        x = self.layers(x)
        x = x.view(len(x), -1)
        return self.head(x) / x.size(-1)


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
    workers: int,
):
    """
    Return a *list* of train DataLoaders created by splitting the CIFAR-100
    training set into `num_splits` **contiguous, in-order** chunks, plus one
    shared test DataLoader. No randomness is used in the split itself.
    """
    if args.datatype.lower() == 'cifar100':
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
    elif args.datatype.lower() == 'cifar10':
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        full_train_set = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=train_transform
        )
        test_set = torchvision.datasets.CIFAR10(
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


# def train_one_epoch(model, loader, optimizer, scaler, device, criterion):
#     model.train()
#     running_loss = 0.0
#     top1_m = 0.0
#     top5_m = 0.0
#     for images, targets in tqdm(loader, desc='Train', leave=False):
#         images = images.to(device, non_blocking=True)
#         targets = targets.to(device, non_blocking=True)

#         optimizer.zero_grad(set_to_none=True)
#         if scaler is not None:
#             with torch.cuda.amp.autocast():
#                 outputs = model(images)
#                 loss = criterion(outputs, targets)
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#         else:
#             outputs = model(images)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()

#         running_loss += loss.item() * images.size(0)
#         t1, t5 = accuracy(outputs, targets, topk=(1, 5))
#         top1_m += t1 * images.size(0)
#         top5_m += t5 * images.size(0)

#     n = len(loader.dataset)
#     return running_loss / n, top1_m / n, top5_m / n


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
            t1, t5 = accuracy(outputs, targets, topk=(1, 5))
            top1_m += t1 * images.size(0)
            top5_m += t5 * images.size(0)

    # n = len(loader.dataset)
    return running_loss / n, top1_m / n, top5_m / n


def save_checkpoint(state: dict, is_best: bool, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(state, os.path.join(out_dir, 'last.pth'))
    if is_best:
        torch.save(state, os.path.join(out_dir, 'best.pth'))



class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, nesterov=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                if 'momentum_buffer' not in state.keys():
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group['nesterov'] else buf

                p.data.mul_(len(p.data)**0.5 / p.data.norm()) # normalize the weight
                update = zeropower_via_newtonschulz5(g.reshape(len(g), -1)).view(g.shape) # whiten the update
                p.data.add_(update, alpha=-lr) # take a step

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
    parser.add_argument('--datatype', type=str)
    parser.add_argument('--modeltype', type=str)
    # args = parser.parse_args([])
    
    # args = parser.parse_args(['--n_workers','8','--train_batch_size','128','--eval_batch_size','3500','--network','ring','--alg','muon','--epochs','200'])

    args = parser.parse_args(['--n_workers','1','--train_batch_size','128','--eval_batch_size','3500','--network','','--alg','stdcenmuon','--epochs','8', '--log_interval','50','--datatype','cifar10', '--modeltype','cifarnet'])

    # args = parser.parse_args(['--n_workers','8','--train_batch_size','64','--eval_batch_size','3500','--network','ring','--alg','gt_nsgdm','--epochs','100', '--log_interval','100'])


    wandb.init(project="my-kaggle-project", name=f'{args.alg}_{args.network}_vitsmall')
    artifact = wandb.Artifact("my_model", type="model")
    
    os.makedirs("graphs", exist_ok=True)
    os.makedirs(f"output/{args.network}", exist_ok=True)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.modeltype=='cifarnet':
        batch_size = 2000
        val_loader = CifarLoader('cifar10', train=False, batch_size=2000)
        train_loader = CifarLoader('cifar10', train=True, batch_size=batch_size, aug=dict(flip=True, translate=2))
        train_images = train_loader.normalize(train_loader.images[:5000])
        loaders = [train_loader]
    else:
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
    if args.modeltype=='cifarnet':
        base_model = CifarNet()
        base_model.reset()
        base_model.init_whiten(train_images)
    elif args.modeltype=='vitsmall':
        base_model = vit_small()
    else:
        pass
    for _ in range(args.n_workers):
        if args.modeltype=='cifarnet':
            m = CifarNet()
        elif args.modeltype=='vitsmall':
            m = vit_small()
        else:
            pass
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
    elif args.alg == 'stdcenmuon':
        batch_size = 2000
        bias_lr = 0.053
        head_lr = 0.67
        wd = 2e-6 * batch_size
        model=workers[0]
        filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
        norm_biases = [p for n, p in model.named_parameters() if 'norm' in n and p.requires_grad]
        param_configs = [dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd/bias_lr),
                        dict(params=norm_biases, lr=bias_lr, weight_decay=wd/bias_lr),
                        dict(params=[model.head.weight], lr=head_lr, weight_decay=wd/head_lr)]
        optimizer1 = torch.optim.SGD(param_configs, momentum=0.85, nesterov=True, fused=True)
        optimizer2 = Muon(filter_params, lr=0.24, momentum=0.6, nesterov=True)
        optimizers = [optimizer1, optimizer2]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]

        

    # if args.alg == 'adamw':
    #     # what's the warmup strategy for adamw?
    #     optimizer = optim.AdamW(workers[0].parameters(), lr=1e-3, weight_decay=0.05, betas=(0.9, 0.999), eps=1e-8)
        
    total_params = sum(p.numel() for p in workers[0].parameters())
    jwp(f"total_params = {total_params}")
    jwp(args.label_smoothing)
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
    whiten_bias_train_steps = 3 * max_round_per_epoch
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
            if args.alg == 'stdcenmuon':
                logits = model(x, whiten_bias_grad=(r < whiten_bias_train_steps))
            else:
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
            elif args.alg == 'stdcenmuon':
                for group in optimizer1.param_groups[:1]:
                    group["lr"] = group["initial_lr"] * (1 - r / whiten_bias_train_steps)
                for group in optimizer1.param_groups[1:]+optimizer2.param_groups:
                    group["lr"] = group["initial_lr"] * (1 - r / total_rounds)
                for opt in optimizers:
                    opt.step()
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
