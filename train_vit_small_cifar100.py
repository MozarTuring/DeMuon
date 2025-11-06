import argparse
import math
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from timm.models.vision_transformer import VisionTransformer

# ----------------------------
# Model from your snippet
# ----------------------------
def vit_small(img_size=32, num_classes=100, drop_path_rate=0.1):
    model = VisionTransformer(
        img_size=img_size,
        patch_size=img_size // 8,  # 4 for 32x32
        in_chans=3,
        num_classes=num_classes,
        embed_dim=384,
        depth=12,
        num_heads=6,
        drop_path_rate=drop_path_rate,
    )
    return model

# ----------------------------
# Utilities
# ----------------------------
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        B = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k * (100.0 / B)).item())
        return res

# Warmup wrapper for any scheduler
class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, after_scheduler, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # linear warmup from 0 -> base lr
            return [base_lr * (self.last_epoch + 1) / float(self.warmup_epochs)
                    for base_lr in self.base_lrs]
        return self.after_scheduler.get_lr()

    def step(self, epoch=None):
        if self.last_epoch < self.warmup_epochs:
            super().step(epoch)
        else:
            if not self.finished:
                # align the inner scheduler's last_epoch
                self.after_scheduler.base_lrs = [g['lr'] for g in self.optimizer.param_groups]
                self.finished = True
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.warmup_epochs)
            self.last_epoch = (self.last_epoch + 1) if epoch is None else epoch

# ----------------------------
# Training / Eval loops
# ----------------------------
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, log_interval=100, grad_clip=None):
    model.train()
    running_loss = 0.0
    running_top1 = 0.0

    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        top1, = accuracy(outputs, targets, topk=(1,))
        running_loss += loss.item() * images.size(0)
        running_top1 += top1 * images.size(0)

        if (step + 1) % log_interval == 0:
            print(f"  [Epoch {epoch:03d}] Step {step+1:04d}/{len(loader)} "
                  f"Loss: {loss.item():.4f} | Top1: {top1:.2f}%")

    n = len(loader.dataset)
    return running_loss / n, running_top1 / n

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16):
            outputs = model(images)
            loss = criterion(outputs, targets)
        top1, = accuracy(outputs, targets, topk=(1,))
        total_loss += loss.item() * images.size(0)
        total_top1 += top1 * images.size(0)
    n = len(loader.dataset)
    return total_loss / n, total_top1 / n

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train ViT-small on CIFAR-100 (from scratch)")
    parser.add_argument("--data", type=str, default="./data", help="root to store CIFAR-100")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.addArgument = parser.add_argument
    parser.add_argument("--lr", type=float, default=5e-4, help="base learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="./checkpoints_vit_cifar100")
    parser.add_argument("--resume", type=str, default="", help="path to checkpoint to resume")
    parser.add_argument("--compile", action="store_true", help="use torch.compile if available")
    args = parser.parse_args([])

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ------------------------
    # Data
    # ------------------------
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    train_set = datasets.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    test_set  = datasets.CIFAR100(root=args.data, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # ------------------------
    # Model, loss, opt, sched
    # ------------------------
    model = vit_small(img_size=32, num_classes=100, drop_path_rate=args.drop_path)
    model.to(device)

    # optional compile for speed (PyTorch 2.0+)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore

    # Label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # AdamW (standard for ViT)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))

    # Cosine schedule with warmup
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.warmup_epochs))
    scheduler = WarmupScheduler(optimizer, warmup_epochs=args.warmup_epochs, after_scheduler=cosine)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ------------------------
    # Resume (optional)
    # ------------------------
    start_epoch = 0
    best_acc = 0.0
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("best_acc", 0.0)
        print(f"Resumed from {args.resume} at epoch {start_epoch} (best_acc={best_acc:.2f}%)")

    # ------------------------
    # Train
    # ------------------------
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs} — lr: {optimizer.param_groups[0]['lr']:.6f}")

        train_loss, train_top1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch+1,
            log_interval=100, grad_clip=args.grad_clip
        )
        val_loss, val_top1 = evaluate(model, test_loader, criterion, device)

        print(f"  Train  — Loss: {train_loss:.4f} | Top1: {train_top1:.2f}%")
        print(f"  Valid  — Loss: {val_loss:.4f} | Top1: {val_top1:.2f}%")

        # step the scheduler AFTER evaluation (epoch-level)
        scheduler.step()

        # Save checkpoint
        is_best = val_top1 > best_acc
        best_acc = max(best_acc, val_top1)
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_acc": best_acc,
            "args": vars(args),
        }
        torch.save(ckpt, save_dir / "last.pth")
        if is_best:
            torch.save(ckpt, save_dir / "best.pth")
            print(f"  ✔ New best: {best_acc:.2f}% (checkpoint saved)")

    print(f"Training complete. Best Top-1: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
