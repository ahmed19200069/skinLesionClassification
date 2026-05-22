"""
Ablation Study — at least 3 controlled experiments varying one factor at a time.

Experiments (all use MobileNetV2 full_finetune strategy as the base):
  EXP-1  Learning rate sensitivity     : lr ∈ {1e-2, 1e-3, 1e-4, 1e-5}
  EXP-2  Dropout rate sensitivity      : dropout ∈ {0.2, 0.35, 0.5, 0.65}
  EXP-3  Data augmentation ablation    : none | flip_only | full_augment
  EXP-4  Class-imbalance strategy      : no_handling | class_weights | sampler | both

Each experiment trains for --ablation_epochs (default 15) — short enough for CPU,
long enough to show relative differences.

Usage:
    python src/ablation.py --experiment lr          --ablation_epochs 15
    python src/ablation.py --experiment dropout     --ablation_epochs 15
    python src/ablation.py --experiment augmentation --ablation_epochs 15
    python src/ablation.py --experiment imbalance   --ablation_epochs 15
    python src/ablation.py --experiment all         --ablation_epochs 15
"""

from __future__ import annotations
import argparse
import copy
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score

from dataset import (
    HAM10000Dataset, get_dataloaders, make_weighted_sampler,
    compute_class_weights, IMAGENET_MEAN, IMAGENET_STD,
    IMG_SIZE, NUM_CLASSES, CLASS_NAMES,
)
from model import SkinLesionMobileNetV2, model_info


# ---------------------------------------------------------------------------
# Mini train/eval loop (shared across all experiments)
# ---------------------------------------------------------------------------

def _run_epochs(model, train_loader, val_loader, criterion,
                optimizer, scheduler, device, epochs) -> list[dict]:
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        tr_correct, tr_total = 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            tr_correct += (out.argmax(1) == labels).sum().item()
            tr_total   += imgs.size(0)
        scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        val_f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        val_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

        history.append({
            "epoch":   epoch,
            "train_acc": round(tr_correct / tr_total, 4),
            "val_acc":   round(val_acc, 4),
            "val_f1":    round(val_f1,  4),
        })
        print(f"  Ep {epoch:02d}/{epochs} | "
              f"tr_acc={tr_correct/tr_total*100:.1f}% | "
              f"val_acc={val_acc*100:.1f}% | val_f1={val_f1:.3f}")
    return history


def _fresh_model(strategy: str = "full_finetune") -> SkinLesionMobileNetV2:
    from model import build_model
    return build_model(strategy)


# ---------------------------------------------------------------------------
# EXP-1  Learning rate sensitivity
# ---------------------------------------------------------------------------

def exp_learning_rate(args, device, train_loader, val_loader, class_weights):
    """Vary LR; everything else fixed. Shows how sensitive MobileNetV2 is to LR."""
    lrs      = [1e-2, 1e-3, 1e-4, 1e-5]
    results  = {}

    for lr in lrs:
        label = f"lr={lr}"
        print(f"\n  [{label}]")
        model     = _fresh_model().to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.ablation_epochs)
        history   = _run_epochs(model, train_loader, val_loader, criterion,
                                optimizer, scheduler, device, args.ablation_epochs)
        results[label] = {
            "lr":         lr,
            "best_val_f1": max(h["val_f1"] for h in history),
            "history":    history,
        }

    return results


# ---------------------------------------------------------------------------
# EXP-2  Dropout sensitivity
# ---------------------------------------------------------------------------

def exp_dropout(args, device, train_loader, val_loader, class_weights):
    """Vary dropout in the classification head; backbone learning rate fixed at 1e-4."""
    dropouts = [0.2, 0.35, 0.5, 0.65]
    results  = {}

    for dp in dropouts:
        label = f"dropout={dp}"
        print(f"\n  [{label}]")
        model     = SkinLesionMobileNetV2(num_classes=NUM_CLASSES).to(device)

        # Rebuild head with custom dropout
        from model import _build_head
        in_feat   = 1280
        model.head = _build_head(in_feat, NUM_CLASSES).to(device)
        # Patch dropout layers in the head with the experiment value
        for module in model.head.modules():
            if isinstance(module, nn.Dropout):
                module.p = dp

        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.ablation_epochs)
        history   = _run_epochs(model, train_loader, val_loader, criterion,
                                optimizer, scheduler, device, args.ablation_epochs)
        results[label] = {
            "dropout":     dp,
            "best_val_f1": max(h["val_f1"] for h in history),
            "history":     history,
        }

    return results


# ---------------------------------------------------------------------------
# EXP-3  Data augmentation ablation
# ---------------------------------------------------------------------------

def _make_transform_variant(variant: str):
    base_resize = transforms.Resize((IMG_SIZE, IMG_SIZE))
    to_tensor   = transforms.ToTensor()
    normalize   = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    if variant == "none":
        return transforms.Compose([base_resize, to_tensor, normalize])

    if variant == "flip_only":
        return transforms.Compose([
            base_resize,
            transforms.RandomHorizontalFlip(),
            to_tensor, normalize,
        ])

    # full_augment (same as production)
    return transforms.Compose([
        transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        to_tensor, normalize,
    ])


def exp_augmentation(args, device, val_loader, class_weights,
                     train_ids, csv_path, img_dirs):
    """Remove/reduce augmentation to quantify its contribution."""
    variants = ["none", "flip_only", "full_augment"]
    results  = {}

    for variant in variants:
        label = f"aug={variant}"
        print(f"\n  [{label}]")
        transform    = _make_transform_variant(variant)
        train_ds     = HAM10000Dataset(csv_path, img_dirs, transform=transform, ids=train_ids)
        sampler      = make_weighted_sampler(train_ds.labels, NUM_CLASSES)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=sampler, num_workers=args.num_workers,
                                  pin_memory=True, drop_last=True)

        model     = _fresh_model().to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.ablation_epochs)
        history   = _run_epochs(model, train_loader, val_loader, criterion,
                                optimizer, scheduler, device, args.ablation_epochs)
        results[label] = {
            "augmentation": variant,
            "best_val_f1":  max(h["val_f1"] for h in history),
            "history":      history,
        }

    return results


# ---------------------------------------------------------------------------
# EXP-4  Class-imbalance handling strategy
# ---------------------------------------------------------------------------

def exp_imbalance(args, device, class_weights, val_loader,
                  train_ids, csv_path, img_dirs):
    """Compare four ways of handling the HAM10000 class imbalance."""
    from torchvision import transforms as T
    tr = _make_transform_variant("full_augment")

    strategies = {
        "no_handling":   (False, None),
        "class_weights": (False, "weights"),
        "sampler":       (True,  None),
        "both":          (True,  "weights"),
    }
    results = {}

    for name, (use_sampler, loss_mode) in strategies.items():
        label = f"imbalance={name}"
        print(f"\n  [{label}]")

        train_ds = HAM10000Dataset(csv_path, img_dirs, transform=tr, ids=train_ids)
        if use_sampler:
            sampler      = make_weighted_sampler(train_ds.labels, NUM_CLASSES)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                      sampler=sampler, num_workers=args.num_workers,
                                      pin_memory=True, drop_last=True)
        else:
            train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.num_workers,
                                      pin_memory=True, drop_last=True)

        cw        = class_weights.to(device) if loss_mode == "weights" else None
        criterion = nn.CrossEntropyLoss(weight=cw)
        model     = _fresh_model().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.ablation_epochs)
        history   = _run_epochs(model, train_loader, val_loader, criterion,
                                optimizer, scheduler, device, args.ablation_epochs)
        results[label] = {
            "use_sampler":  use_sampler,
            "loss_weights": loss_mode == "weights",
            "best_val_f1":  max(h["val_f1"] for h in history),
            "history":      history,
        }

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ablation(results: dict, title: str, x_key: str, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for label, data in results.items():
        hist   = data["history"]
        epochs = [h["epoch"]   for h in hist]
        val_f1 = [h["val_f1"]  for h in hist]
        val_acc= [h["val_acc"] for h in hist]
        axes[0].plot(epochs, val_f1,  marker="o", markersize=3, label=label)
        axes[1].plot(epochs, val_acc, marker="o", markersize=3, label=label)

    axes[0].set_title(f"Val Macro F1 — {title}")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Macro F1")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    axes[1].set_title(f"Val Accuracy — {title}")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {out_path}")


def print_summary(exp_name: str, results: dict):
    print(f"\n{'─'*50}")
    print(f"  {exp_name} — Summary")
    print(f"{'─'*50}")
    for label, data in results.items():
        print(f"  {label:30s} best_val_f1 = {data['best_val_f1']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="all",
                        choices=["lr", "dropout", "augmentation", "imbalance", "all"])
    parser.add_argument("--csv",         default="data/HAM10000_metadata.csv")
    parser.add_argument("--img_dirs",    nargs="+",
                        default=["data/HAM10000_images_part1",
                                 "data/HAM10000_images_part2"])
    parser.add_argument("--ablation_epochs", type=int, default=15)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--val_split",   type=float, default=0.2)
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Experiment: {args.experiment}")

    # Build shared data split
    train_loader, val_loader, class_weights = get_dataloaders(
        args.csv, args.img_dirs,
        val_split=args.val_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Extract train_ids for augmentation / imbalance experiments
    import pandas as pd, numpy as np
    meta     = pd.read_csv(args.csv)
    rng      = np.random.default_rng(42)
    val_ids  = []
    from dataset import CLASS_NAMES as CLS
    for cls in CLS:
        ids  = meta.loc[meta["dx"] == cls, "image_id"].tolist()
        n    = max(1, int(len(ids) * args.val_split))
        val_ids.extend(rng.choice(ids, size=n, replace=False).tolist())
    val_set   = set(val_ids)
    train_ids = [iid for iid in meta["image_id"] if iid not in val_set]

    all_results = {}

    # ── EXP-1 LR ──────────────────────────────────────────────────────────
    if args.experiment in ("lr", "all"):
        print("\n" + "="*55)
        print("EXP-1: Learning Rate Sensitivity")
        res = exp_learning_rate(args, device, train_loader, val_loader, class_weights)
        print_summary("EXP-1 LR", res)
        plot_ablation(res, "EXP-1: Learning Rate Sensitivity", "lr",
                      os.path.join(args.results_dir, "ablation_lr.png"))
        all_results["exp1_lr"] = res

    # ── EXP-2 DROPOUT ─────────────────────────────────────────────────────
    if args.experiment in ("dropout", "all"):
        print("\n" + "="*55)
        print("EXP-2: Dropout Rate Sensitivity")
        res = exp_dropout(args, device, train_loader, val_loader, class_weights)
        print_summary("EXP-2 Dropout", res)
        plot_ablation(res, "EXP-2: Dropout Sensitivity", "dropout",
                      os.path.join(args.results_dir, "ablation_dropout.png"))
        all_results["exp2_dropout"] = res

    # ── EXP-3 AUGMENTATION ────────────────────────────────────────────────
    if args.experiment in ("augmentation", "all"):
        print("\n" + "="*55)
        print("EXP-3: Data Augmentation Ablation")
        res = exp_augmentation(args, device, val_loader, class_weights,
                               train_ids, args.csv, args.img_dirs)
        print_summary("EXP-3 Augmentation", res)
        plot_ablation(res, "EXP-3: Augmentation Ablation", "aug",
                      os.path.join(args.results_dir, "ablation_augmentation.png"))
        all_results["exp3_augmentation"] = res

    # ── EXP-4 IMBALANCE ───────────────────────────────────────────────────
    if args.experiment in ("imbalance", "all"):
        print("\n" + "="*55)
        print("EXP-4: Class Imbalance Strategy")
        res = exp_imbalance(args, device, class_weights, val_loader,
                            train_ids, args.csv, args.img_dirs)
        print_summary("EXP-4 Imbalance", res)
        plot_ablation(res, "EXP-4: Class Imbalance Strategy", "imbalance",
                      os.path.join(args.results_dir, "ablation_imbalance.png"))
        all_results["exp4_imbalance"] = res

    # ── Save all results ───────────────────────────────────────────────────
    out = os.path.join(args.results_dir, "ablation_results.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll ablation results saved: {out}")


if __name__ == "__main__":
    main()
