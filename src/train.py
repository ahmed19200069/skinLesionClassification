"""
Training script for all three MobileNetV2 strategies.

Usage:
  # Strategy 1 - feature extraction (backbone frozen)
  python src/train.py --strategy feature_extraction --epochs 20

  # Strategy 2 - progressive unfreezing
  python src/train.py --strategy progressive --epochs 30 \
      --unfreeze5_epoch 10 --unfreeze10_epoch 20

  # Strategy 3 - full fine-tuning
  python src/train.py --strategy full_finetune --epochs 30 --lr 1e-4
"""

from __future__ import annotations
import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, precision_score, recall_score

from dataset import get_dataloaders, CLASS_NAMES, NUM_CLASSES
from model import build_model, model_info


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(1)
        correct   += (preds == labels).sum().item()
        total     += imgs.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    acc = correct / total
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    pre = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / total, acc, f1, pre, rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy",    required=True,
                        choices=["feature_extraction", "progressive", "full_finetune"])
    parser.add_argument("--csv",         default="data/HAM10000_metadata.csv")
    parser.add_argument("--img_dirs",    nargs="+",
                        default=["data/HAM10000_images_part1",
                                 "data/HAM10000_images_part2"])
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--lr_backbone", type=float, default=1e-4,
                        help="Lower LR for backbone layers when unfrozen")
    parser.add_argument("--weight_decay",type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int,   default=2)
    parser.add_argument("--val_split",   type=float, default=0.2)
    parser.add_argument("--ckpt_dir",    default="checkpoints")
    parser.add_argument("--results_dir", default="results")
    # progressive unfreezing schedule
    parser.add_argument("--unfreeze5_epoch",  type=int, default=10,
                        help="Epoch to unfreeze last 5 backbone blocks")
    parser.add_argument("--unfreeze10_epoch", type=int, default=20,
                        help="Epoch to unfreeze last 10 backbone blocks")
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir,    exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device   : {device}")
    print(f"Strategy : {args.strategy}")

    train_loader, val_loader, class_weights = get_dataloaders(
        args.csv, args.img_dirs,
        val_split=args.val_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(args.strategy).to(device)
    info  = model_info(model)
    print(f"Trainable params: {info['trainable_params']:,} / {info['total_params']:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    def make_optimizer(model):
        # separate param groups so the backbone can use a lower lr than the head
        head_params     = list(model.head.parameters()) + list(model.pool.parameters())
        backbone_params = [p for p in model.features.parameters() if p.requires_grad]
        groups = [{"params": head_params, "lr": args.lr}]
        if backbone_params:
            groups.append({"params": backbone_params, "lr": args.lr_backbone})
        return optim.Adam(groups, weight_decay=args.weight_decay)

    optimizer = make_optimizer(model)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_f1 = 0.0
    ckpt_path = os.path.join(args.ckpt_dir, f"{args.strategy}_best.pth")

    for epoch in range(1, args.epochs + 1):
        # progressive unfreezing: rebuild optimizer whenever we unlock more layers
        if args.strategy == "progressive":
            if epoch == args.unfreeze5_epoch:
                print(f"  [Epoch {epoch}] Unfreezing last 5 backbone blocks")
                model.unfreeze_last_n_blocks(5)
                optimizer = make_optimizer(model)
                scheduler = CosineAnnealingLR(
                    optimizer, T_max=args.epochs - epoch + 1)
            elif epoch == args.unfreeze10_epoch:
                print(f"  [Epoch {epoch}] Unfreezing last 10 backbone blocks")
                model.unfreeze_last_n_blocks(10)
                optimizer = make_optimizer(model)
                scheduler = CosineAnnealingLR(
                    optimizer, T_max=args.epochs - epoch + 1)

        t0 = time.perf_counter()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, va_f1, va_pre, va_rec = val_epoch(
            model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.perf_counter() - t0

        row = {
            "epoch": epoch,
            "train_loss": round(tr_loss, 4), "train_acc": round(tr_acc, 4),
            "val_loss":   round(va_loss, 4), "val_acc":   round(va_acc, 4),
            "val_f1":     round(va_f1,  4),
            "val_precision": round(va_pre, 4),
            "val_recall":    round(va_rec, 4),
        }
        history.append(row)

        print(
            f"Ep {epoch:03d}/{args.epochs} | "
            f"Tr {tr_loss:.4f}/{tr_acc*100:.1f}% | "
            f"Va {va_loss:.4f}/{va_acc*100:.1f}% | "
            f"F1={va_f1:.3f} | {elapsed:.1f}s"
        )

        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save({"model_state": model.state_dict(),
                        "strategy":    args.strategy,
                        "epoch":       epoch,
                        "val_f1":      va_f1}, ckpt_path)

    # save final weights too, not just the best
    torch.save({"model_state": model.state_dict(), "strategy": args.strategy},
               os.path.join(args.ckpt_dir, f"{args.strategy}_final.pth"))

    summary = {
        "strategy":       args.strategy,
        "best_val_f1":    best_f1,
        "model_info":     info,
        "hyperparameters": vars(args),
        "history":        history,
    }
    out = os.path.join(args.results_dir, f"{args.strategy}_train.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBest val F1 : {best_f1:.4f}")
    print(f"Checkpoint  : {ckpt_path}")
    print(f"Results     : {out}")


if __name__ == "__main__":
    main()
