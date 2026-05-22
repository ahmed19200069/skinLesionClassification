"""
Week 4: Load the three eval JSON files and produce a side-by-side comparison
table + bar charts.

Usage:
    python src/compare.py --results_dir results
"""

from __future__ import annotations
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STRATEGIES = ["baseline_cnn", "feature_extraction", "progressive", "full_finetune"]
STRATEGY_LABELS = {
    "baseline_cnn":       "Baseline CNN\n(from scratch)",
    "feature_extraction": "Feature Extraction\n(frozen backbone)",
    "progressive":        "Progressive\nUnfreezing",
    "full_finetune":      "Full Fine-Tune",
}
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_eval(results_dir: str, strategy: str) -> dict | None:
    path = os.path.join(results_dir, f"{strategy}_eval.json")
    if not os.path.isfile(path):
        print(f"  [warn] missing {path}")
        return None
    with open(path) as f:
        return json.load(f)


def load_train(results_dir: str, strategy: str) -> dict | None:
    path = os.path.join(results_dir, f"{strategy}_train.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Summary table (printed + saved as txt)
# ---------------------------------------------------------------------------

def print_summary_table(data: dict[str, dict], out_dir: str):
    metrics = ["accuracy", "macro_f1", "weighted_f1", "macro_precision",
               "macro_recall", "auc_ovr"]
    col_w = 22

    lines = []
    header = f"{'Metric':<20}" + "".join(
        f"{STRATEGY_LABELS[s].replace(chr(10),' '):>{col_w}}" for s in STRATEGIES if s in data
    )
    lines.append(header)
    lines.append("-" * len(header))

    for m in metrics:
        row = f"{m:<20}"
        for s in STRATEGIES:
            if s not in data:
                continue
            v = data[s].get(m)
            row += f"{(f'{v:.4f}' if v is not None else 'N/A'):>{col_w}}"
        lines.append(row)

    # BKL→MEL danger row
    lines.append("-" * len(header))
    row = f"{'mel→bkl (danger %)':<20}"
    for s in STRATEGIES:
        if s not in data:
            continue
        pct = data[s].get("bkl_mel_confusion", {}).get("mel_misclassified_as_bkl_pct", "N/A")
        row += f"{(str(pct)+'%'):>{col_w}}"
    lines.append(row)

    row = f"{'bkl→mel (%)':<20}"
    for s in STRATEGIES:
        if s not in data:
            continue
        pct = data[s].get("bkl_mel_confusion", {}).get("bkl_misclassified_as_mel_pct", "N/A")
        row += f"{(str(pct)+'%'):>{col_w}}"
    lines.append(row)

    text = "\n".join(lines)
    print("\n" + text)
    path = os.path.join(out_dir, "comparison_table.txt")
    with open(path, "w") as f:
        f.write(text + "\n")
    print(f"\nSaved: {path}")


# ---------------------------------------------------------------------------
# Bar chart: overall metrics
# ---------------------------------------------------------------------------

def plot_overall_metrics(data: dict[str, dict], out_dir: str):
    metrics     = ["accuracy", "macro_f1", "weighted_f1", "macro_recall"]
    metric_lbls = ["Accuracy", "Macro F1", "Weighted F1", "Macro Recall"]
    present     = [s for s in STRATEGIES if s in data]
    x           = np.arange(len(metrics))
    n           = len(present)
    width       = 0.8 / max(n, 1)
    colors      = ["#95a5a6", "#3498db", "#e67e22", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, strat in enumerate(present):
        vals = [data[strat].get(m, 0) for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=strat.replace("_", " ").title(),
                      color=colors[i], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_lbls, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Strategy Comparison — Overall Metrics", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "comparison_overall.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Grouped bar: per-class F1
# ---------------------------------------------------------------------------

def plot_per_class_f1(data: dict[str, dict], out_dir: str):
    present = [s for s in STRATEGIES if s in data]
    x       = np.arange(len(CLASS_NAMES))
    n       = len(present)
    width   = 0.8 / max(n, 1)
    colors  = ["#95a5a6", "#3498db", "#e67e22", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, strat in enumerate(present):
        vals = [data[strat].get("per_class_f1", {}).get(c, 0) for c in CLASS_NAMES]
        ax.bar(x + i * width, vals, width,
               label=strat.replace("_", " ").title(),
               color=colors[i], alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 by Strategy\n(mel & bkl often confused)", fontsize=13)
    ax.legend(fontsize=10)
    ax.axvspan(CLASS_NAMES.index("bkl") - 0.3,
               CLASS_NAMES.index("bkl") + 0.9, alpha=0.07, color="red",
               label="bkl-mel confusion zone")
    ax.axvspan(CLASS_NAMES.index("mel") - 0.3,
               CLASS_NAMES.index("mel") + 0.9, alpha=0.07, color="red")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "comparison_per_class_f1.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Learning curves (from train JSONs)
# ---------------------------------------------------------------------------

def plot_learning_curves(results_dir: str, out_dir: str):
    colors = {"baseline_cnn":       "#95a5a6",
              "feature_extraction": "#3498db",
              "progressive":        "#e67e22",
              "full_finetune":      "#2ecc71"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for strat in STRATEGIES:
        td = load_train(results_dir, strat)
        if td is None:
            continue
        hist   = td.get("history", [])
        epochs = [h["epoch"]     for h in hist]
        tr_acc = [h["train_acc"] for h in hist]
        va_acc = [h["val_acc"]   for h in hist]
        va_f1  = [h["val_f1"]    for h in hist]
        lbl    = strat.replace("_", " ").title()
        c      = colors[strat]
        axes[0].plot(epochs, tr_acc, "--", color=c, alpha=0.6, label=f"{lbl} (train)")
        axes[0].plot(epochs, va_acc, "-",  color=c,            label=f"{lbl} (val)")
        axes[1].plot(epochs, va_f1,  "-",  color=c,            label=lbl)

    axes[0].set_title("Accuracy Curves")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].set_title("Validation Macro F1")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Macro F1")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    plt.suptitle("Learning Curves — All Strategies", fontsize=14)
    plt.tight_layout()
    path = os.path.join(out_dir, "learning_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# BKL–MEL danger chart
# ---------------------------------------------------------------------------

def plot_bkl_mel(data: dict[str, dict], out_dir: str):
    present = [s for s in STRATEGIES if s in data]
    mel_pct = [data[s]["bkl_mel_confusion"]["mel_misclassified_as_bkl_pct"] for s in present]
    bkl_pct = [data[s]["bkl_mel_confusion"]["bkl_misclassified_as_mel_pct"] for s in present]
    lbls    = [s.replace("_", "\n").title() for s in present]
    x       = np.arange(len(present))
    width   = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, mel_pct, width, label="mel→bkl (missed melanoma)", color="#e74c3c")
    ax.bar(x + width/2, bkl_pct, width, label="bkl→mel (false alarm)",     color="#f39c12")
    ax.set_xticks(x)
    ax.set_xticklabels(lbls, fontsize=10)
    ax.set_ylabel("% of class mis-classified")
    ax.set_title("BKL ↔ MEL Confusion by Strategy\n(mel→bkl = clinically dangerous)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "bkl_mel_confusion.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    data = {}
    for s in STRATEGIES:
        d = load_eval(args.results_dir, s)
        if d:
            data[s] = d

    if not data:
        print("No eval JSON files found. Run evaluate.py for each strategy first.")
        return

    print_summary_table(data, args.results_dir)
    plot_overall_metrics(data, args.results_dir)
    plot_per_class_f1(data, args.results_dir)
    plot_learning_curves(args.results_dir, args.results_dir)
    plot_bkl_mel(data, args.results_dir)

    # Save combined JSON
    out = os.path.join(args.results_dir, "all_strategies_comparison.json")
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nCombined comparison saved: {out}")


if __name__ == "__main__":
    main()
