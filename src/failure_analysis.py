"""
Failure mode analysis for the Week 4 report section.

Questions I'm trying to answer:
  1. Which images does each strategy most consistently get wrong?
  2. Are the failures clustered in specific classes (especially bkl/mel)?
  3. Do image brightness or contrast correlate with errors?
  4. Which samples are wrong across ALL strategies (truly hard cases)?

Usage:
    python src/failure_analysis.py \
        --strategies feature_extraction progressive full_finetune \
        --top_k 20
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
from PIL import Image, ImageStat

from dataset import (
    HAM10000Dataset, get_dataloaders, CLASS_NAMES, NUM_CLASSES,
    get_val_transform,
)
from model import build_model
from baseline_cnn import BaselineCNN


@torch.no_grad()
def collect_predictions(model, dataset: HAM10000Dataset, device, batch_size=32):
    """Returns arrays: true_labels, pred_labels, confidences, image_paths."""
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_true, all_pred, all_conf = [], [], []

    model.eval()
    for imgs, labels in loader:
        imgs    = imgs.to(device)
        logits  = model(imgs)
        probs   = torch.softmax(logits, dim=1).cpu().numpy()
        preds   = probs.argmax(axis=1)
        confs   = probs.max(axis=1)
        all_true.extend(labels.numpy())
        all_pred.extend(preds)
        all_conf.extend(confs)

    return (
        np.array(all_true),
        np.array(all_pred),
        np.array(all_conf),
        dataset.paths,
    )


def image_stats(path: str) -> dict:
    """Get mean brightness and contrast (std) from grayscale image."""
    try:
        img  = Image.open(path).convert("L")
        stat = ImageStat.Stat(img)
        return {
            "brightness": round(stat.mean[0], 2),
            "contrast":   round(stat.stddev[0], 2),
        }
    except Exception:
        return {"brightness": None, "contrast": None}


def per_class_error_rate(true, pred) -> dict:
    rates = {}
    for i, cls in enumerate(CLASS_NAMES):
        mask  = true == i
        if mask.sum() == 0:
            rates[cls] = None
            continue
        rates[cls] = round(float((pred[mask] != i).mean()), 4)
    return rates


def top_k_hardest(true, pred, paths, k=20) -> list[dict]:
    """Returns the first k wrong predictions (wrong index is a simple proxy for difficulty)."""
    wrong = np.where(true != pred)[0]
    return [
        {"idx": int(i), "true": CLASS_NAMES[true[i]],
         "pred": CLASS_NAMES[pred[i]], "path": paths[i]}
        for i in wrong[:k]
    ]


def consensus_failures(per_strategy: dict[str, np.ndarray],
                       true: np.ndarray,
                       paths: list[str]) -> list[dict]:
    """Finds images that are wrong in every evaluated strategy."""
    n          = len(true)
    wrong_sets = [set(np.where(true != pred)[0].tolist())
                  for pred in per_strategy.values()]
    common     = wrong_sets[0].intersection(*wrong_sets[1:]) if len(wrong_sets) > 1 else wrong_sets[0]
    return [
        {"idx": int(i), "true": CLASS_NAMES[true[i]], "path": paths[i],
         "predictions": {s: CLASS_NAMES[p[i]] for s, p in per_strategy.items()}}
        for i in sorted(common)
    ]


def plot_error_rates(per_strategy_errors: dict[str, dict], out_dir: str):
    strategies = list(per_strategy_errors.keys())
    x          = np.arange(len(CLASS_NAMES))
    width      = 0.8 / max(len(strategies), 1)
    colors     = ["#95a5a6", "#3498db", "#e67e22", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, strat in enumerate(strategies):
        vals = [per_strategy_errors[strat].get(c) or 0 for c in CLASS_NAMES]
        ax.bar(x + i * width, vals, width,
               label=strat.replace("_", " ").title(),
               color=colors[i % len(colors)], alpha=0.85)

    ax.set_xticks(x + width * (len(strategies) - 1) / 2)
    ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.set_ylabel("Error Rate (fraction wrong)")
    ax.set_title("Per-Class Error Rate by Strategy", fontsize=13)
    ax.legend(fontsize=9)
    # shade the bkl and mel columns - those are the problematic ones
    ax.axvspan(CLASS_NAMES.index("bkl") - 0.4,
               CLASS_NAMES.index("bkl") + 0.9, alpha=0.07, color="red")
    ax.axvspan(CLASS_NAMES.index("mel") - 0.4,
               CLASS_NAMES.index("mel") + 0.9, alpha=0.07, color="red")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "failure_per_class_errors.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_failure_grid(failures: list[dict], title: str, out_path: str, n=12):
    """Display a grid of misclassified images."""
    failures = failures[:n]
    cols     = 4
    rows     = (len(failures) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()

    for ax in axes:
        ax.axis("off")

    for ax, info in zip(axes, failures):
        try:
            img = Image.open(info["path"]).convert("RGB").resize((112, 112))
            ax.imshow(img)
        except Exception:
            ax.set_facecolor("#ddd")
        true_lbl = info["true"]
        pred_lbl = info.get("pred", "?")
        ax.set_title(f"T:{true_lbl}\nP:{pred_lbl}", fontsize=8,
                     color="red" if true_lbl != pred_lbl else "green")

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_brightness_vs_error(true, pred, paths, strategy: str, out_dir: str):
    """Check whether darker or lower-contrast images fail more often."""
    correct_b, correct_c = [], []
    wrong_b,   wrong_c   = [], []

    for i, (t, p, path) in enumerate(zip(true, pred, paths)):
        stats = image_stats(path)
        if stats["brightness"] is None:
            continue
        if t == p:
            correct_b.append(stats["brightness"])
            correct_c.append(stats["contrast"])
        else:
            wrong_b.append(stats["brightness"])
            wrong_c.append(stats["contrast"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(correct_b, bins=30, alpha=0.6, label="Correct",  color="#2ecc71")
    axes[0].hist(wrong_b,   bins=30, alpha=0.6, label="Wrong",    color="#e74c3c")
    axes[0].set_xlabel("Mean Brightness")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Brightness Distribution")
    axes[0].legend()

    axes[1].hist(correct_c, bins=30, alpha=0.6, label="Correct",  color="#2ecc71")
    axes[1].hist(wrong_c,   bins=30, alpha=0.6, label="Wrong",    color="#e74c3c")
    axes[1].set_xlabel("Contrast (std of grayscale)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Contrast Distribution")
    axes[1].legend()

    plt.suptitle(f"Visual Properties vs Failure — {strategy}", fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, f"failure_brightness_{strategy}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategies", nargs="+",
                        default=["baseline_cnn", "feature_extraction",
                                 "progressive", "full_finetune"])
    parser.add_argument("--csv",         default="data/HAM10000_metadata.csv")
    parser.add_argument("--img_dirs",    nargs="+",
                        default=["data/HAM10000_images_part1",
                                 "data/HAM10000_images_part2"])
    parser.add_argument("--val_split",   type=float, default=0.2)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--top_k",       type=int,   default=20,
                        help="Number of hardest failures to visualise")
    parser.add_argument("--ckpt_dir",    default="checkpoints")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build val set without augmentation
    _, val_loader, _ = get_dataloaders(
        args.csv, args.img_dirs,
        val_split=args.val_split,
        batch_size=args.batch_size,
        num_workers=0,
        use_sampler=False,
    )
    val_ds = val_loader.dataset

    per_strategy_preds  = {}
    per_strategy_errors = {}
    true_labels         = None
    image_paths         = val_ds.paths

    for strat in args.strategies:
        ckpt_path = os.path.join(args.ckpt_dir, f"{strat}_best.pth")
        if not os.path.isfile(ckpt_path):
            print(f"  [skip] No checkpoint: {ckpt_path}")
            continue

        if strat == "baseline_cnn":
            model = BaselineCNN(num_classes=NUM_CLASSES).to(device)
        else:
            model = build_model(strat).to(device)

        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"\nAnalysing failures: {strat}")

        true, pred, conf, paths = collect_predictions(model, val_ds, device, args.batch_size)
        if true_labels is None:
            true_labels = true

        per_strategy_preds[strat]  = pred
        per_strategy_errors[strat] = per_class_error_rate(true, pred)

        hard = top_k_hardest(true, pred, paths, k=args.top_k)
        plot_failure_grid(
            hard, f"Top-{args.top_k} Failures — {strat}",
            os.path.join(args.results_dir, f"failure_grid_{strat}.png"),
            n=min(12, args.top_k),
        )

        plot_brightness_vs_error(true, pred, paths, strat, args.results_dir)

    if not per_strategy_preds:
        print("No checkpoints found. Train models first.")
        return

    # find images that are wrong in every strategy
    consensus = consensus_failures(per_strategy_preds, true_labels, image_paths)
    print(f"\nConsensus failures (wrong in ALL strategies): {len(consensus)}")
    if consensus:
        plot_failure_grid(
            [{"path": c["path"], "true": c["true"], "pred": "ALL_WRONG"}
             for c in consensus[:12]],
            f"Consensus Failures (wrong in all {len(per_strategy_preds)} strategies)",
            os.path.join(args.results_dir, "failure_consensus.png"),
        )

    if per_strategy_errors:
        plot_error_rates(per_strategy_errors, args.results_dir)

    report = {
        "per_class_error_rates": per_strategy_errors,
        "consensus_failure_count": len(consensus),
        "consensus_failures_sample": consensus[:20],
    }
    out = os.path.join(args.results_dir, "failure_analysis.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFailure analysis saved: {out}")


if __name__ == "__main__":
    main()
