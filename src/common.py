"""
common.py - shared paths, label constants, metric helpers and plot styling.

Why: both routing approaches (traditional ML and the zero-shot decision model)
must be evaluated with exactly the same flag convention and the same metric
code, otherwise the comparison would not be fair. Everything that both sides
share lives here.
"""
from pathlib import Path
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
DATASET_CSV = DATA_DIR / "messages.csv"
GROUPS_CSV = DATA_DIR / "message_groups.csv"

# Routing flag convention required by the assignment.
SUPPORT, SALES = 0, 1
FLAG_NAMES = {SUPPORT: "support", SALES: "sales"}
# Decision-model label -> routing flag.
DECISION_TO_FLAG = {"customer_service": SUPPORT, "sales": SALES}

SEED = 42

# Plot styling (validated categorical slots 1-2 of the dataviz reference palette).
COLOR_SUPPORT = "#2a78d6"
COLOR_SALES = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e4e3df"


def ensure_dirs():
    for d in (DATA_DIR, RESULTS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def compute_metrics(y_true, y_pred):
    """Metrics with sales (flag 1) as the positive class, plus macro F1."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[SUPPORT, SALES])
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_sales": float(precision_score(y_true, y_pred, pos_label=SALES, zero_division=0)),
        "recall_sales": float(recall_score(y_true, y_pred, pos_label=SALES, zero_division=0)),
        "f1_sales": float(f1_score(y_true, y_pred, pos_label=SALES, zero_division=0)),
        "precision_support": float(precision_score(y_true, y_pred, pos_label=SUPPORT, zero_division=0)),
        "recall_support": float(recall_score(y_true, y_pred, pos_label=SUPPORT, zero_division=0)),
        "f1_support": float(f1_score(y_true, y_pred, pos_label=SUPPORT, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        # rows = true (support, sales), cols = predicted (support, sales)
        "confusion_matrix": cm.tolist(),
    }


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def plot_confusion_matrix(cm, title, path):
    """Single-hue (blue) sequential heatmap with counts written in each cell."""
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue_seq", ["#eef4fc", "#2a78d6", "#0d3b73"])
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.2, 3.6), facecolor=SURFACE)
    sns.heatmap(cm, annot=False, cmap=cmap, cbar=False, linewidths=2, linecolor=SURFACE,
                vmin=0, vmax=max(cm.max(), 1), ax=ax, square=True)
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            dark_cell = v > 0.55 * max(cm.max(), 1)
            ax.text(j + 0.5, i + 0.5, str(v), ha="center", va="center", fontsize=16,
                    color="white" if dark_cell else INK)
    ax.set_xticklabels(["support (0)", "sales (1)"])
    ax.set_yticklabels(["support (0)", "sales (1)"], rotation=0)
    ax.set_xlabel("Predicted flag")
    ax.set_ylabel("True flag")
    ax.set_title(title, fontsize=11)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
