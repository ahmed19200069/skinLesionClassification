"""
Evaluation script - runs the saved model on the validation set and reports metrics.
Also looks specifically at bkl/mel confusion because those two classes are visually
similar and mel->bkl errors (missed melanoma) are the most dangerous kind of mistake.

Usage:
    python src/evaluate.py --strategy baseline_cnn
    python src/evaluate.py --strategy feature_extraction
    python src/evaluate.py --strategy progressive
    python src/evaluate.py --strategy full_finetune
"""

from __future__ import annotations
import argparse
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from dataset import get_dataloaders, CLASS_NAMES, NUM_CLASSES
from model import build_model
from baseline_cnn import BaselineCNN


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs  = torch.softmax(logits, dim=1).cpu()
        preds  = logits.argmax(1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        all_probs.append(probs.numpy())
    return (
        np.array(all_labels),
        np.array(all_preds),
        np.vstack(all_probs),
    )


def plot_confusion_matrix(cm: np.ndarray, strategy: str, out_dir: str):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True",      fontsize=12)
    ax.set_title(f"Confusion Matrix — {strategy}", fontsize=14)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{strategy}_confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_per_class_f1(per_class_f1: dict, strategy: str, out_dir: str):
    names  = list(per_class_f1.keys())
    values = list(per_class_f1.values())
    # highlight mel and bkl in red since they're the problem pair
    colors = ["#e74c3c" if n in ("mel", "bkl") else "#3498db" for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, values, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 Score")
    ax.set_title(f"Per-Class F1 — {strategy}\n(red = mel & bkl confusion pair)")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{strategy}_per_class_f1.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def bkl_mel_analysis(labels, preds, cm) -> dict:
    mel_idx = CLASS_NAMES.index("mel")
    bkl_idx = CLASS_NAMES.index("bkl")

    mel_as_bkl = cm[mel_idx, bkl_idx]   # true=mel, pred=bkl  <- dangerous
    bkl_as_mel = cm[bkl_idx, mel_idx]   # true=bkl, pred=mel  <- false alarm

    total_mel = cm[mel_idx].sum()
    total_bkl = cm[bkl_idx].sum()

    return {
        "mel_misclassified_as_bkl": int(mel_as_bkl),
        "mel_misclassified_as_bkl_pct": round(mel_as_bkl / max(total_mel, 1) * 100, 1),
        "bkl_misclassified_as_mel": int(bkl_as_mel),
        "bkl_misclassified_as_mel_pct": round(bkl_as_mel / max(total_bkl, 1) * 100, 1),
        "note": (
            "mel->bkl errors are clinically dangerous (missed melanoma). "
            "bkl->mel errors cause unnecessary procedures."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True,
                        choices=["baseline_cnn", "feature_extraction",
                                 "progressive", "full_finetune"])
    parser.add_argument("--csv",      default="data/HAM10000_metadata.csv")
    parser.add_argument("--img_dirs", nargs="+",
                        default=["data/HAM10000_images_part1",
                                 "data/HAM10000_images_part2"])
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--val_split",   type=float, default=0.2)
    parser.add_argument("--ckpt_dir",    default="checkpoints")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader, _ = get_dataloaders(
        args.csv, args.img_dirs,
        val_split=args.val_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_sampler=False,
    )

    if args.strategy == "baseline_cnn":
        model = BaselineCNN(num_classes=NUM_CLASSES).to(device)
    else:
        model = build_model(args.strategy).to(device)
    ckpt_path = os.path.join(args.ckpt_dir, f"{args.strategy}_best.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint: {ckpt_path}  (epoch {ckpt.get('epoch','?')})")

    labels, preds, probs = get_predictions(model, val_loader, device)

    acc  = (preds == labels).mean()
    f1_macro  = f1_score(labels, preds, average="macro",    zero_division=0)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)
    pre  = precision_score(labels, preds, average="macro",   zero_division=0)
    rec  = recall_score(labels, preds, average="macro",      zero_division=0)

    # AUC - one-vs-rest
    labels_bin = label_binarize(labels, classes=list(range(NUM_CLASSES)))
    try:
        auc = roc_auc_score(labels_bin, probs, average="macro", multi_class="ovr")
    except ValueError:
        auc = None

    per_f1_vals = f1_score(labels, preds, average=None, zero_division=0)
    per_class_f1 = {cls: round(float(v), 4)
                    for cls, v in zip(CLASS_NAMES, per_f1_vals)}

    cm = confusion_matrix(labels, preds)
    bkl_mel = bkl_mel_analysis(labels, preds, cm)

    print(f"\n{'='*55}")
    print(f"Strategy : {args.strategy}")
    print(f"Accuracy : {acc*100:.2f}%")
    print(f"Macro F1 : {f1_macro:.4f}")
    print(f"W. F1    : {f1_weighted:.4f}")
    print(f"Precision: {pre:.4f}  Recall: {rec:.4f}")
    if auc:
        print(f"AUC (OvR): {auc:.4f}")
    print(f"\nPer-class F1:")
    for cls, v in per_class_f1.items():
        marker = " <- check" if cls in ("mel", "bkl") else ""
        print(f"  {cls:6s} : {v:.4f}{marker}")
    print(f"\nBKL <-> MEL confusion:")
    print(f"  mel->bkl: {bkl_mel['mel_misclassified_as_bkl']} "
          f"({bkl_mel['mel_misclassified_as_bkl_pct']}% of mel)")
    print(f"  bkl->mel: {bkl_mel['bkl_misclassified_as_mel']} "
          f"({bkl_mel['bkl_misclassified_as_mel_pct']}% of bkl)")
    print(f"  {bkl_mel['note']}")
    print(classification_report(labels, preds, target_names=CLASS_NAMES, zero_division=0))

    plot_confusion_matrix(cm, args.strategy, args.results_dir)
    plot_per_class_f1(per_class_f1, args.strategy, args.results_dir)

    result = {
        "strategy":     args.strategy,
        "accuracy":     round(float(acc),        4),
        "macro_f1":     round(float(f1_macro),   4),
        "weighted_f1":  round(float(f1_weighted), 4),
        "macro_precision": round(float(pre),      4),
        "macro_recall":    round(float(rec),      4),
        "auc_ovr":      round(float(auc), 4) if auc else None,
        "per_class_f1": per_class_f1,
        "bkl_mel_confusion": bkl_mel,
        "confusion_matrix": cm.tolist(),
    }
    out = os.path.join(args.results_dir, f"{args.strategy}_eval.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nEvaluation saved: {out}")


if __name__ == "__main__":
    main()
