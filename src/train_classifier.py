"""
train_classifier.py - traditional ML router (approach 1).

What it does
  1. Builds two numeric representations of a message:
       * TF-IDF  - weights of word n-grams (lower-cased, sublinear tf, L2 norm).
                   Stop words are kept, because "not/no/never" carry the
                   negation cases.
       * HAND    - the 11 interpretable numeric features of features.py.
     and their concatenation (UNION).
  2. Searches over the model families the brief allows - linear SVM, degree-2
     polynomial-kernel SVM, and logistic regression on degree-2 polynomial
     features - with 5-fold StratifiedGroupKFold cross-validation on the 40
     TRAINING messages only (paraphrase pairs stay inside one fold, so a model
     is never scored on the near-duplicate of a message it trained on).
  3. Refits the single selected configuration on all 40 training messages and
     predicts the 20 test messages exactly once.
  4. Saves metrics, per-message predictions, errors, feature weights and three
     figures, including the decision boundary in a 2D feature projection.

Outputs (results/): ml_cv_results.csv, ml_metrics.json, ml_predictions.csv,
  ml_errors.csv, ml_feature_importance.csv, ml_confusion_matrix.png,
  ml_decision_boundary.png, ml_decision_boundary_lexicon.png,
  ml_test_scores.png ; models/ml_router.joblib
"""
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC

from common import (DATASET_CSV, GROUPS_CSV, RESULTS_DIR, MODELS_DIR, SEED, SUPPORT, SALES,
                    COLOR_SUPPORT, COLOR_SALES, INK, INK_2, SURFACE, GRID,
                    compute_metrics, ensure_dirs, plot_confusion_matrix, save_json, style_axes)
from features import HandcraftedFeatures, FEATURE_NAMES, featurize

C_GRID = [0.01, 0.1, 1, 10, 100]
N_FOLDS = 5


def feature_step(feature_set):
    """feature_set -> (list of pipeline steps before the classifier, is_sparse)."""
    tfidf1 = ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 1), sublinear_tf=True))
    tfidf2 = ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True))
    hand = [("hand", HandcraftedFeatures()), ("scale", StandardScaler())]
    if feature_set == "tfidf_1_1":
        return [tfidf1], True
    if feature_set == "tfidf_1_2":
        return [tfidf2], True
    if feature_set == "hand":
        return hand, False
    if feature_set == "union":
        return [("union", FeatureUnion([
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 1), sublinear_tf=True)),
            ("hand", Pipeline(hand)),
        ]))], True
    raise ValueError(feature_set)


def classifier_step(model, C):
    """model name -> the final estimator step(s), as the brief allows."""
    if model == "svm_linear":
        return [("clf", SVC(kernel="linear", C=C))]
    if model == "svm_poly2":
        return [("clf", SVC(kernel="poly", degree=2, coef0=1.0, gamma="scale", C=C))]
    if model == "logreg_poly2":
        return [("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("clf", LogisticRegression(C=C, max_iter=5000, random_state=SEED))]
    raise ValueError(model)


def make_pipeline(feature_set, model, C):
    steps, _ = feature_step(feature_set)
    return Pipeline(list(steps) + classifier_step(model, C))


def build_grid():
    grid = []
    for feature_set in ("hand", "tfidf_1_1", "tfidf_1_2", "union"):
        for model in ("svm_linear", "svm_poly2", "logreg_poly2"):
            # degree-2 polynomial EXPANSION of a TF-IDF space is intractable here
            # (thousands of features -> millions of products), so it is only run
            # on the 11 hand-crafted features. The poly *kernel* SVM covers the
            # non-linear case for the sparse text representations.
            if model == "logreg_poly2" and feature_set != "hand":
                continue
            for C in C_GRID:
                grid.append({"feature_set": feature_set, "model": model, "C": C})
    return grid


def run_cv(train, grid):
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rows = []
    for p in grid:
        scores = cross_val_score(make_pipeline(**p), train.message, train.label,
                                 groups=train.group_id, cv=cv, scoring="accuracy")
        rows.append({**p, "cv_mean_accuracy": float(scores.mean()),
                     "cv_std_accuracy": float(scores.std()),
                     "fold_scores": " ".join(f"{s:.3f}" for s in scores)})
        print(f"  {p['feature_set']:10s} {p['model']:13s} C={p['C']:<6g} "
              f"CV acc {scores.mean():.3f} +/- {scores.std():.3f}")
    return pd.DataFrame(rows)


def plot_boundary(Z_train, y_train, Z_test, y_test, clf2d, test_ids,
                  xlabel, ylabel, title, path):
    allZ = np.vstack([Z_train, Z_test])
    padx = 0.06 * (allZ[:, 0].max() - allZ[:, 0].min() + 1e-9)
    pady = 0.06 * (allZ[:, 1].max() - allZ[:, 1].min() + 1e-9)
    xx, yy = np.meshgrid(
        np.linspace(allZ[:, 0].min() - padx, allZ[:, 0].max() + padx, 400),
        np.linspace(allZ[:, 1].min() - pady, allZ[:, 1].max() + pady, 400))
    dec = clf2d.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7.8, 5.8), facecolor=SURFACE)
    ax.contourf(xx, yy, dec, levels=[dec.min() - 1, 0, dec.max() + 1],
                colors=["#dce9f8", "#fbe2d7"], alpha=0.85)
    ax.contour(xx, yy, dec, levels=[0], colors=[INK], linewidths=2)
    ax.contour(xx, yy, dec, levels=[-1, 1], colors=[INK_2], linewidths=1, linestyles="--")

    col = lambda y: [COLOR_SUPPORT if v == SUPPORT else COLOR_SALES for v in y]
    ax.scatter(Z_train[:, 0], Z_train[:, 1], c=col(y_train), marker="o", s=55,
               edgecolors=SURFACE, linewidths=2, zorder=3)
    ax.scatter(Z_test[:, 0], Z_test[:, 1], c=col(y_test), marker="^", s=95,
               edgecolors=INK, linewidths=1, zorder=4)
    for (x, yv), t, p, mid in zip(Z_test, y_test, clf2d.predict(Z_test), test_ids):
        if t != p:
            ax.scatter([x], [yv], s=270, facecolors="none", edgecolors=INK, linewidths=1.5, zorder=5)
            ax.annotate(f"id {mid}", (x, yv), xytext=(8, 7), textcoords="offset points",
                        fontsize=8, color=INK)

    handles = [
        Line2D([], [], marker="o", ls="", color=COLOR_SUPPORT, markersize=8, label="support (0) - train"),
        Line2D([], [], marker="o", ls="", color=COLOR_SALES, markersize=8, label="sales (1) - train"),
        Line2D([], [], marker="^", ls="", color=COLOR_SUPPORT, mec=INK, markersize=9, label="support (0) - test"),
        Line2D([], [], marker="^", ls="", color=COLOR_SALES, mec=INK, markersize=9, label="sales (1) - test"),
        Line2D([], [], marker="o", ls="", mfc="none", mec=INK, markersize=13, label="misclassified test"),
        Line2D([], [], color=INK, lw=2, label="decision boundary"),
        Line2D([], [], color=INK_2, lw=1, ls="--", label="margin (score = +/-1)"),
    ]
    ax.legend(handles=handles, loc="best", fontsize=8, frameon=True, facecolor=SURFACE, edgecolor=GRID)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_test_scores(pred_df, path):
    d = pred_df.sort_values("decision_score").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.8, 6.6), facecolor=SURFACE)
    ax.barh(range(len(d)), d.decision_score,
            color=[COLOR_SUPPORT if t == SUPPORT else COLOR_SALES for t in d.label], height=0.7)
    ax.axvline(0, color=INK, lw=1.5)
    labels = [f"[{r.id}] {r.message if len(r.message) <= 58 else r.message[:55] + '...'}"
              f"{'  X' if r.label != r.pred_flag else ''}" for _, r in d.iterrows()]
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Classifier decision score  (< 0 -> support flag 0,  > 0 -> sales flag 1)")
    ax.set_title("Traditional ML router: decision scores on the 20 test messages\n"
                 "bar colour = true label (blue support, orange sales); X = misrouted", fontsize=10)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def permutation_importance_table(model, train, path):
    """Model-agnostic feature importance for the selected model.

    The selected classifier may be a kernel SVM, which has no linear weights, so
    importance is measured by permutation: shuffle one transformed feature column
    at a time and record how much training accuracy drops.
    """
    from sklearn.inspection import permutation_importance
    pre = Pipeline(model.steps[:-1]) if "poly" not in model.named_steps else Pipeline(model.steps[:-2])
    X = pre.transform(train.message)
    if sparse.issparse(X) or X.shape[1] != len(FEATURE_NAMES):
        return None       # only meaningful for the 11 named hand-crafted features
    clf = Pipeline(model.steps[len(pre.steps):])
    r = permutation_importance(clf, X, train.label, n_repeats=50, random_state=SEED,
                               scoring="accuracy")
    df = (pd.DataFrame({"feature": FEATURE_NAMES,
                        "importance_mean_accuracy_drop": r.importances_mean,
                        "importance_std": r.importances_std})
          .sort_values("importance_mean_accuracy_drop", ascending=False))
    df.to_csv(path, index=False)
    return df


def linear_weights_table(model, feature_set, path):
    """Write the most influential features, when the fitted model is linear."""
    clf = model.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return None
    coef = np.ravel(clf.coef_.toarray() if sparse.issparse(clf.coef_) else clf.coef_)
    if feature_set == "hand" and "poly" in model.named_steps:
        names = model.named_steps["poly"].get_feature_names_out(FEATURE_NAMES)
    elif feature_set == "hand":
        names = np.array(FEATURE_NAMES)
    elif feature_set == "union":
        names = np.concatenate([
            model.named_steps["union"].transformer_list[0][1].get_feature_names_out(),
            np.array(FEATURE_NAMES)])
    else:
        names = model.named_steps["tfidf"].get_feature_names_out()
    names = np.asarray(names)
    if len(names) != len(coef):
        return None
    order = np.argsort(coef)
    k = min(15, len(coef) // 2)
    df = pd.concat([
        pd.DataFrame({"feature": names[order[:k]], "weight": coef[order[:k]], "pushes_to": "support (0)"}),
        pd.DataFrame({"feature": names[order[::-1][:k]], "weight": coef[order[::-1][:k]], "pushes_to": "sales (1)"}),
    ])
    df.to_csv(path, index=False)
    return df


def main():
    ensure_dirs()
    df = pd.read_csv(DATASET_CSV).merge(pd.read_csv(GROUPS_CSV), on="id")
    train = df[df.split == "train"].reset_index(drop=True)
    test = df[df.split == "test"].reset_index(drop=True)

    # ---- 1. model + feature-set selection, training split only ----
    grid = build_grid()
    print(f"Grouped {N_FOLDS}-fold CV over {len(grid)} configurations (train only):")
    cv_df = run_cv(train, grid)
    cv_df.to_csv(RESULTS_DIR / "ml_cv_results.csv", index=False)
    best_i = int(np.argmax(cv_df.cv_mean_accuracy.values))   # ties -> earliest = simplest
    best = grid[best_i]
    print("\nBest per feature set:", cv_df.groupby("feature_set").cv_mean_accuracy.max().round(3).to_dict())
    print("Selected:", best, f"(CV acc {cv_df.cv_mean_accuracy.iloc[best_i]:.3f})")

    # ---- 2. refit on the full training split, predict the test split once ----
    model = make_pipeline(**best).fit(train.message, train.label)
    joblib.dump(model, MODELS_DIR / "ml_router.joblib")

    t0 = time.perf_counter()
    pred = model.predict(test.message)
    latency_ms = (time.perf_counter() - t0) * 1000 / len(test)
    scores = model.decision_function(test.message)
    train_acc = float((model.predict(train.message) == train.label).mean())
    metrics = compute_metrics(test.label, pred)

    pred_df = test[["id", "message", "label", "group_id", "category"]].copy()
    pred_df["pred_flag"] = pred
    pred_df["decision_score"] = scores
    pred_df["correct"] = pred_df.pred_flag == pred_df.label
    pred_df.to_csv(RESULTS_DIR / "ml_predictions.csv", index=False)
    pred_df[~pred_df.correct].to_csv(RESULTS_DIR / "ml_errors.csv", index=False)
    per_cat = pred_df.groupby("category").correct.agg(["sum", "count"])

    linear_weights_table(model, best["feature_set"], RESULTS_DIR / "ml_top_features.csv")
    imp = permutation_importance_table(model, train, RESULTS_DIR / "ml_feature_importance.csv")

    # ---- 3a. boundary in a 2D projection of the SELECTED feature space ----
    pre = Pipeline(model.steps[:-1]) if "poly" not in model.named_steps else Pipeline(model.steps[:-2])
    X_train, X_test = pre.transform(train.message), pre.transform(test.message)
    if sparse.issparse(X_train):
        proj = TruncatedSVD(n_components=2, random_state=SEED).fit(X_train)
        proj_name = "TruncatedSVD (LSA)"
    else:
        proj = PCA(n_components=2, random_state=SEED).fit(X_train)
        proj_name = "PCA"
    Z_train, Z_test = proj.transform(X_train), proj.transform(X_test)
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cands = [(c, cross_val_score(SVC(kernel="linear", C=c), Z_train, train.label,
                                 groups=train.group_id, cv=cv).mean()) for c in C_GRID]
    c2d = cands[int(np.argmax([s for _, s in cands]))][0]
    clf2d = SVC(kernel="linear", C=c2d).fit(Z_train, train.label)
    acc2d = {"train": float((clf2d.predict(Z_train) == train.label).mean()),
             "test": float((clf2d.predict(Z_test) == test.label).mean())}
    plot_boundary(Z_train, train.label.values, Z_test, test.label.values, clf2d, test.id.values,
                  f"{proj_name} component 1", f"{proj_name} component 2",
                  f"Linear SVM boundary in a 2D {proj_name} projection of the "
                  f"'{best['feature_set']}' feature space\n"
                  f"(2D model: train acc {acc2d['train']:.2f}, test acc {acc2d['test']:.2f}; "
                  f"the deployed model uses all features)",
                  RESULTS_DIR / "ml_decision_boundary.png")

    # ---- 3b. boundary on two interpretable raw features ----
    i_sup, i_sal = FEATURE_NAMES.index("support_hits"), FEATURE_NAMES.index("sales_hits")
    H_train = featurize(train.message)[:, [i_sup, i_sal]]
    H_test = featurize(test.message)[:, [i_sup, i_sal]]
    cands = [(c, cross_val_score(SVC(kernel="linear", C=c), H_train, train.label,
                                 groups=train.group_id, cv=cv).mean()) for c in C_GRID]
    c_lex = cands[int(np.argmax([s for _, s in cands]))][0]
    clf_lex = SVC(kernel="linear", C=c_lex).fit(H_train, train.label)
    acc_lex = {"train": float((clf_lex.predict(H_train) == train.label).mean()),
               "test": float((clf_lex.predict(H_test) == test.label).mean())}
    plot_boundary(H_train, train.label.values, H_test, test.label.values, clf_lex, test.id.values,
                  "support_hits (support lexicon hits per 10 words)",
                  "sales_hits (sales lexicon hits per 10 words)",
                  "Linear SVM boundary on the two interpretable lexicon features\n"
                  f"(train acc {acc_lex['train']:.2f}, test acc {acc_lex['test']:.2f})",
                  RESULTS_DIR / "ml_decision_boundary_lexicon.png")

    plot_test_scores(pred_df, RESULTS_DIR / "ml_test_scores.png")
    plot_confusion_matrix(metrics["confusion_matrix"],
                          f"Traditional ML router (test, n={len(test)})",
                          RESULTS_DIR / "ml_confusion_matrix.png")

    n_feat = X_train.shape[1]
    save_json({
        "selected": {"feature_set": best["feature_set"], "model": best["model"], "C": best["C"]},
        "selection": f"{N_FOLDS}-fold StratifiedGroupKFold on the training split "
                     f"(groups = paraphrase pairs), accuracy, {len(grid)} configurations",
        "best_cv_mean_accuracy": float(cv_df.cv_mean_accuracy.iloc[best_i]),
        "best_cv_std_accuracy": float(cv_df.cv_std_accuracy.iloc[best_i]),
        "cv_best_per_feature_set": cv_df.groupby("feature_set").cv_mean_accuracy.max().round(4).to_dict(),
        "cv_best_per_model": cv_df.groupby("model").cv_mean_accuracy.max().round(4).to_dict(),
        "n_features": int(n_feat),
        "train_accuracy": train_acc,
        "test": metrics,
        "test_accuracy_by_category": {c: f"{int(r['sum'])}/{int(r['count'])}" for c, r in per_cat.iterrows()},
        "latency_ms_per_message": latency_ms,
        "projection_2d": {"method": f"{proj_name}(2) fitted on the training split, then linear SVC",
                          "C": c2d, **{f"{k}_accuracy": v for k, v in acc2d.items()}},
        "lexicon_2d": {"features": ["support_hits", "sales_hits"], "C": c_lex,
                       **{f"{k}_accuracy": v for k, v in acc_lex.items()}},
    }, RESULTS_DIR / "ml_metrics.json")

    print(f"\nTrain acc {train_acc:.3f} | Test acc {metrics['accuracy']:.3f} | "
          f"macro F1 {metrics['f1_macro']:.3f}")
    print("Confusion matrix (test) [[TN FP],[FN TP]]:", metrics["confusion_matrix"])
    print("Per category (test):", {c: f"{int(r['sum'])}/{int(r['count'])}" for c, r in per_cat.iterrows()})
    print(f"2D projection model: train {acc2d['train']:.3f}, test {acc2d['test']:.3f}")
    print(f"Lexicon 2D model:    train {acc_lex['train']:.3f}, test {acc_lex['test']:.3f}")
    if imp is not None:
        print("Permutation importance (train):\n", imp.to_string(index=False))
    print("Errors:\n", pred_df[~pred_df.correct][["id", "label", "pred_flag", "decision_score", "message"]]
          .to_string(index=False, max_colwidth=60))


if __name__ == "__main__":
    main()
