"""
compare.py - comparative analysis of the two routing approaches.

Reads the saved predictions of both routers (nothing is recomputed except the
small learning curve, which uses the training split only) and produces:
  * a side-by-side metrics table on the 20 test messages;
  * the paired McNemar test on the test split - do the two routers differ by
    more than chance on the messages where they disagree?
  * the agreement matrix and the per-message error overlap;
  * accuracy per message category (plain / negation / mixed);
  * a learning curve for the traditional classifier (grouped CV on the training
    split as the training set grows), which shows how much labelled data the
    trained approach would still need;
  * a post-hoc sensitivity check: how other ML configurations would have scored
    on the test set. These numbers did NOT influence the selection and are
    reported only to show how unstable the choice is at this dataset size.

Outputs (results/): comparison_table.csv, comparison_summary.json,
  error_overlap.csv, comparison_accuracy.png, comparison_by_category.png,
  ml_learning_curve.png, ml_posthoc_sensitivity.csv
"""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

from common import (DATASET_CSV, GROUPS_CSV, RESULTS_DIR, SEED, SUPPORT, SALES,
                    COLOR_SUPPORT, COLOR_SALES, INK, INK_2, SURFACE, GRID,
                    compute_metrics, ensure_dirs, save_json, style_axes)
from train_classifier import make_pipeline, N_FOLDS

ML = "Traditional ML"
DM = "Decision model"
METRIC_ROWS = [
    ("accuracy", "Accuracy"),
    ("precision_sales", "Precision (sales)"),
    ("recall_sales", "Recall (sales)"),
    ("f1_sales", "F1 (sales)"),
    ("precision_support", "Precision (support)"),
    ("recall_support", "Recall (support)"),
    ("f1_support", "F1 (support)"),
    ("f1_macro", "Macro F1"),
]


def mcnemar_exact(b, c):
    """Exact binomial McNemar test on the discordant pairs (two-sided)."""
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5)))


def plot_accuracy(ml_m, dm_m, path):
    labels = [name for _, name in METRIC_ROWS]
    ml_v = [ml_m[k] for k, _ in METRIC_ROWS]
    dm_v = [dm_m[k] for k, _ in METRIC_ROWS]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.6, 5.0), facecolor=SURFACE)
    ax.barh(y - 0.19, ml_v, height=0.34, color=COLOR_SUPPORT, label=ML)
    ax.barh(y + 0.19, dm_v, height=0.34, color=COLOR_SALES, label=DM)
    for yi, v in zip(y - 0.19, ml_v):
        ax.text(v + 0.012, yi, f"{v:.2f}", va="center", fontsize=8, color=INK_2)
    for yi, v in zip(y + 0.19, dm_v):
        ax.text(v + 0.012, yi, f"{v:.2f}", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Score on the 20 test messages")
    ax.set_title("Routing quality: traditional ML classifier vs zero-shot decision model",
                 fontsize=11, pad=26)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.005))
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_by_category(merged, path):
    cats = ["plain", "negation", "mixed"]
    g = merged.groupby("category")
    ml_v = [float(g.get_group(c).ml_correct.mean()) for c in cats]
    dm_v = [float(g.get_group(c).dm_correct.mean()) for c in cats]
    n = [len(g.get_group(c)) for c in cats]
    x = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(6.6, 4.4), facecolor=SURFACE)
    ax.bar(x - 0.19, ml_v, width=0.34, color=COLOR_SUPPORT, label=ML)
    ax.bar(x + 0.19, dm_v, width=0.34, color=COLOR_SALES, label=DM)
    for xi, v in zip(x - 0.19, ml_v):
        ax.text(xi, v + 0.025, f"{v:.2f}", ha="center", fontsize=9, color=INK_2)
    for xi, v in zip(x + 0.19, dm_v):
        ax.text(xi, v + 0.025, f"{v:.2f}", ha="center", fontsize=9, color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={k})" for c, k in zip(cats, n)])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Accuracy on the test split")
    ax.set_title("Where the two routers differ: accuracy by message category", fontsize=11)
    ax.legend(frameon=True, facecolor=SURFACE, edgecolor=GRID, fontsize=9, loc="lower left")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def learning_curve(train, best, dm_test_acc, path):
    """Grouped-CV accuracy of the selected ML config as the training set grows."""
    rng = np.random.RandomState(SEED)
    groups = sorted(train.group_id.unique())
    sup = [g for g in groups if train[train.group_id == g].label.iloc[0] == SUPPORT]
    sal = [g for g in groups if train[train.group_id == g].label.iloc[0] == SALES]
    sizes, means, stds = [], [], []
    for k in range(3, len(sup) + 1):
        accs = []
        for rep in range(8):
            chosen = set(rng.choice(sup, k, replace=False)) | set(rng.choice(sal, k, replace=False))
            sub = train[train.group_id.isin(chosen)]
            cv = StratifiedGroupKFold(n_splits=min(N_FOLDS, k), shuffle=True, random_state=SEED + rep)
            accs.append(cross_val_score(make_pipeline(**best), sub.message, sub.label,
                                        groups=sub.group_id, cv=cv, scoring="accuracy").mean())
        sizes.append(2 * 2 * k)        # groups per class * 2 classes * 2 messages per group
        means.append(float(np.mean(accs)))
        stds.append(float(np.std(accs)))
    means, stds = np.array(means), np.array(stds)

    fig, ax = plt.subplots(figsize=(7.0, 4.4), facecolor=SURFACE)
    ax.fill_between(sizes, means - stds, means + stds, color=COLOR_SUPPORT, alpha=0.15, lw=0)
    ax.plot(sizes, means, color=COLOR_SUPPORT, lw=2, marker="o", ms=5, label=f"{ML} (grouped CV on train)")
    ax.axhline(dm_test_acc, color=COLOR_SALES, lw=2, ls="--",
               label=f"{DM}, zero labelled examples ({dm_test_acc:.2f} on test)")
    ax.set_xlabel("Number of labelled training messages used")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.3, 1.02)
    ax.set_title("How much labelled data the trained classifier still needs", fontsize=11)
    ax.legend(frameon=True, facecolor=SURFACE, edgecolor=GRID, fontsize=8.5, loc="lower right")
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return pd.DataFrame({"n_train_messages": sizes, "cv_accuracy_mean": means, "cv_accuracy_std": stds})


def posthoc_sensitivity(train, test, cv_df, path):
    """Test accuracy of the top CV configurations - post-hoc, NOT used for selection."""
    top = cv_df.sort_values("cv_mean_accuracy", ascending=False).head(6)
    rows = []
    for _, r in top.iterrows():
        p = {"feature_set": r.feature_set, "model": r.model, "C": float(r.C)}
        m = make_pipeline(**p).fit(train.message, train.label)
        rows.append({**p, "cv_mean_accuracy": r.cv_mean_accuracy, "cv_std_accuracy": r.cv_std_accuracy,
                     "test_accuracy": float((m.predict(test.message) == test.label).mean())})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def main():
    ensure_dirs()
    ml_pred = pd.read_csv(RESULTS_DIR / "ml_predictions.csv")
    dm_all = pd.read_csv(RESULTS_DIR / "zeroshot_predictions.csv")
    ml_meta = json.loads((RESULTS_DIR / "ml_metrics.json").read_text(encoding="utf-8"))
    dm_meta = json.loads((RESULTS_DIR / "zeroshot_metrics.json").read_text(encoding="utf-8"))
    dm_pred = dm_all[dm_all.split == "test"]

    merged = ml_pred[["id", "message", "label", "category", "pred_flag", "decision_score", "correct"]].rename(
        columns={"pred_flag": "ml_flag", "decision_score": "ml_score", "correct": "ml_correct"}).merge(
        dm_pred[["id", "pred_flag", "score_sales", "final_decision", "correct"]].rename(
            columns={"pred_flag": "dm_flag", "score_sales": "dm_score_sales",
                     "final_decision": "dm_decision", "correct": "dm_correct"}), on="id")

    ml_m = compute_metrics(merged.label, merged.ml_flag)
    dm_m = compute_metrics(merged.label, merged.dm_flag)

    table = pd.DataFrame({
        "metric": [name for _, name in METRIC_ROWS],
        ML: [round(ml_m[k], 3) for k, _ in METRIC_ROWS],
        DM: [round(dm_m[k], 3) for k, _ in METRIC_ROWS],
    })
    table.to_csv(RESULTS_DIR / "comparison_table.csv", index=False)

    # paired comparison on the test split
    b = int(((merged.ml_correct) & (~merged.dm_correct)).sum())   # ML right, DM wrong
    c = int(((~merged.ml_correct) & (merged.dm_correct)).sum())   # ML wrong, DM right
    p = mcnemar_exact(b, c)
    agreement = float((merged.ml_flag == merged.dm_flag).mean())

    merged["both_correct"] = merged.ml_correct & merged.dm_correct
    merged["only_ml_correct"] = merged.ml_correct & ~merged.dm_correct
    merged["only_dm_correct"] = ~merged.ml_correct & merged.dm_correct
    merged["both_wrong"] = ~merged.ml_correct & ~merged.dm_correct
    merged.to_csv(RESULTS_DIR / "error_overlap.csv", index=False)

    plot_accuracy(ml_m, dm_m, RESULTS_DIR / "comparison_accuracy.png")
    plot_by_category(merged, RESULTS_DIR / "comparison_by_category.png")

    # learning curve + post-hoc sensitivity (training split only for selection)
    df = pd.read_csv(DATASET_CSV).merge(pd.read_csv(GROUPS_CSV), on="id")
    train, test = df[df.split == "train"], df[df.split == "test"]
    best = {"feature_set": ml_meta["selected"]["feature_set"], "model": ml_meta["selected"]["model"],
            "C": ml_meta["selected"]["C"]}
    lc = learning_curve(train, best, dm_m["accuracy"], RESULTS_DIR / "ml_learning_curve.png")
    lc.to_csv(RESULTS_DIR / "ml_learning_curve.csv", index=False)
    sens = posthoc_sensitivity(train, test, pd.read_csv(RESULTS_DIR / "ml_cv_results.csv"),
                               RESULTS_DIR / "ml_posthoc_sensitivity.csv")

    save_json({
        "test_set_size": int(len(merged)),
        "traditional_ml": {"config": ml_meta["selected"], "cv_accuracy": ml_meta["best_cv_mean_accuracy"],
                           "train_accuracy": ml_meta["train_accuracy"], "test": ml_m,
                           "latency_ms_per_message": ml_meta["latency_ms_per_message"],
                           "by_category": ml_meta["test_accuracy_by_category"]},
        "decision_model": {"model": dm_meta["model"], "config": dm_meta["selected_config"],
                           "threshold": dm_meta["decision_threshold"],
                           "train_accuracy": dm_meta["train_accuracy"], "test": dm_m,
                           "accuracy_all_60": dm_meta["all_60"]["accuracy"],
                           "latency_ms_per_message": dm_meta["seconds_per_message"] * 1000,
                           "by_category": dm_meta["test_accuracy_by_category"],
                           "abstentions_test": dm_meta["n_abstentions_test"],
                           "schema_invalid_outputs": dm_meta["schema_invalid_outputs"]},
        "paired_test": {"method": "exact two-sided McNemar on the 20 test messages",
                        "ml_right_dm_wrong": b, "ml_wrong_dm_right": c, "p_value": p,
                        "significant_at_0.05": bool(p < 0.05)},
        "agreement_rate": agreement,
        "both_correct": int(merged.both_correct.sum()),
        "only_ml_correct": int(merged.only_ml_correct.sum()),
        "only_dm_correct": int(merged.only_dm_correct.sum()),
        "both_wrong": int(merged.both_wrong.sum()),
        "learning_curve_last": {"n_train_messages": int(lc.n_train_messages.iloc[-1]),
                                "cv_accuracy_mean": float(lc.cv_accuracy_mean.iloc[-1])},
        "posthoc_sensitivity_note": "test accuracies of the top-6 CV configurations; reported "
                                    "after the fact, they played no part in selecting the model",
        "posthoc_test_accuracy_range": [float(sens.test_accuracy.min()), float(sens.test_accuracy.max())],
    }, RESULTS_DIR / "comparison_summary.json")

    print(table.to_string(index=False))
    print(f"\nAgreement {agreement:.2f} | both correct {int(merged.both_correct.sum())} | "
          f"only ML {b} | only DM {c} | both wrong {int(merged.both_wrong.sum())}")
    print(f"McNemar exact p = {p:.4f} ({'significant' if p < 0.05 else 'not significant'} at 0.05)")
    print("\nPost-hoc sensitivity (top CV configs, test accuracy):")
    print(sens.to_string(index=False))
    print("\nLearning curve tail:\n", lc.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
