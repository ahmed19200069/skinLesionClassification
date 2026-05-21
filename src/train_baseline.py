"""
Week 1 - trains the from-scratch baseline CNN on HAM10000.

Usage:
    python src/train_baseline.py --epochs 40 --batch_size 32
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

from dataset import get_dataloaders, NUM_CLASSES
from baseline_cnn import BaselineCNN, model_info


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
        preds       = logits.argmax(1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    acc = correct / total
    f1  = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
    pre = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="macro",    zero_division=0)
    return total_loss / total, acc, f1, pre, rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",         default="data/HAM10000_metadata.csv")
    parser.add_argument("--img_dirs",    nargs="+",
                        default=["data/HAM10000_images_part1",
                                 "data/HAM10000_images_part2"])
    parser.add_argument("--epochs",      type=int,   default=40)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--weight_decay",type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int,   default=2)
    parser.add_argument("--val_split",   type=float, default=0.2)
    parser.add_argument("--ckpt_dir",    default="checkpoints")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir,    exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device   : {device}")
    print(f"Strategy : baseline_cnn (scratch)")

    train_loader, val_loader, class_weights = get_dataloaders(
        args.csv, args.img_dirs,
        val_split=args.val_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model     = BaselineCNN(num_classes=NUM_CLASSES).to(device)
    info      = model_info(model)
    print(f"Parameters: {info['total_params']:,} | Size: {info['size_mb']} MB")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    history   = []
    best_f1   = 0.0
    ckpt_path = os.path.join(args.ckpt_dir, "baseline_cnn_best.pth")

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, va_f1, va_pre, va_rec = val_epoch(
            model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.perf_counter() - t0

        row = {
            "epoch":     epoch,
            "train_loss": round(tr_loss, 4), "train_acc": round(tr_acc, 4),
            "val_loss":   round(va_loss, 4), "val_acc":   round(va_acc, 4),
            "val_f1":     round(va_f1,   4),
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
                        "strategy":    "baseline_cnn",
                        "epoch":       epoch,
                        "val_f1":      va_f1}, ckpt_path)

    torch.save({"model_state": model.state_dict(), "strategy": "baseline_cnn"},
               os.path.join(args.ckpt_dir, "baseline_cnn_final.pth"))

    summary = {
        "strategy":        "baseline_cnn",
        "best_val_f1":     best_f1,
        "model_info":      info,
        "hyperparameters": vars(args),
        "history":         history,
    }
    out = os.path.join(args.results_dir, "baseline_cnn_train.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBest val F1 : {best_f1:.4f}")
    print(f"Checkpoint  : {ckpt_path}")
    print(f"Results     : {out}")


if __name__ == "__main__":
    main()
