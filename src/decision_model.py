"""
decision_model.py - decision-model router: zero-shot NLI with a structured-output wrapper.

Stands in for a commercial Decision API (e.g. Jev by TypeSafe AI). Instead of a
paid service we run a locally cached open-source model and wrap it so the output
is a validated, machine-readable decision object - the property that makes a
Decision API usable inside a pipeline.

Model    facebook/bart-large-mnli (407M params, BART-large fine-tuned on MultiNLI)
License  MIT (model weights); the MultiNLI training corpus is distributed under
         its own permissive terms - see the model card.
Setup    pre-downloaded into the local HuggingFace cache; run with HF_HUB_OFFLINE=1,
         CPU only, no API key, no network call, no training on our data.
Method   Zero-shot classification by natural-language inference: the message is
         the premise, each candidate label is verbalised into a hypothesis via a
         template, and the entailment scores of the two hypotheses are softmaxed
         against each other.

Routing instructions = the hypothesis template plus the two label verbalisations.
This is the open-source equivalent of the "instructions" field of a Decision API.
Four instruction variants and the decision threshold are selected on the 40
TRAINING messages only; the 20 test messages are scored once with the selection.

Structured-output wrapper (what a Decision API would provide):
  * parse raw scores into {"decision", "confidence", "scores"};
  * validate: decision must be in the allowed enum, confidence a float in [0,1],
    scores must cover exactly the allowed labels and sum to ~1, and the
    confidence must equal the top score. Otherwise OutputValidationError;
  * abstain inside a narrow band around the decision threshold: the wrapper
    returns "uncertain" and the router falls back to customer_service (flag 0),
    so an unsure sales guess never strands a customer who has a real problem;
  * map decision -> routing flag (customer_service=0, sales=1).

Outputs (results/): zeroshot_scores_all_configs.csv, zeroshot_config_selection.csv,
  zeroshot_predictions.csv, zeroshot_metrics.json, zeroshot_errors.csv,
  zeroshot_confusion_matrix.png, zeroshot_confidence.png
"""
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import (DATASET_CSV, GROUPS_CSV, RESULTS_DIR, SUPPORT, SALES, DECISION_TO_FLAG,
                    COLOR_SUPPORT, COLOR_SALES, INK, SURFACE, GRID,
                    compute_metrics, ensure_dirs, plot_confusion_matrix, save_json, style_axes)

MODEL_NAME = "facebook/bart-large-mnli"
ALLOWED = ("customer_service", "sales")
ABSTAIN_BAND = 0.05                       # +/- around the decision threshold
FALLBACK_DECISION = "customer_service"    # safe default

# Candidate routing instructions. Each = (name, hypothesis_template, {label: verbalisation}).
CONFIGS = [
    ("A_short_team", "This message should be handled by the {} team.",
     {"customer_service": "customer service", "sales": "sales"}),
    ("B_topic", "This customer message is about {}.",
     {"customer_service": "a problem with a service the customer already has",
      "sales": "buying a new product, a price or an upgrade"}),
    ("C_long_routing", "This customer message should be routed to {}.",
     {"customer_service": ("the customer service team, because it reports a problem, a complaint, "
                           "a billing error or a request about an existing service"),
      "sales": ("the sales team, because it asks about buying a new product, a price, a quote, "
                "an upgrade or a service package")}),
    ("D_intent", "The customer's intent is {}.",
     {"customer_service": "to get help with an existing problem",
      "sales": "to buy something or ask about prices and packages"}),
]
THRESHOLD_GRID = np.round(np.arange(0.20, 0.81, 0.01), 2)


class OutputValidationError(ValueError):
    """Raised when the model's structured output does not satisfy the schema."""


def validate_decision_object(obj):
    """Schema check for the decision object. Returns it unchanged or raises."""
    if not isinstance(obj, dict):
        raise OutputValidationError("decision object is not a mapping")
    for key in ("decision", "confidence", "scores"):
        if key not in obj:
            raise OutputValidationError(f"missing field: {key}")
    if obj["decision"] not in ALLOWED:
        raise OutputValidationError(f"decision {obj['decision']!r} not in {ALLOWED}")
    conf = obj["confidence"]
    if not isinstance(conf, float) or not (0.0 <= conf <= 1.0):
        raise OutputValidationError(f"confidence {conf!r} is not a float in [0, 1]")
    scores = obj["scores"]
    if not isinstance(scores, dict) or set(scores) != set(ALLOWED):
        raise OutputValidationError(f"scores keys {sorted(scores)} != {sorted(ALLOWED)}")
    if abs(sum(scores.values()) - 1.0) > 1e-3:
        raise OutputValidationError(f"scores sum to {sum(scores.values()):.4f}, expected 1.0")
    if abs(scores[obj["decision"]] - conf) > 1e-6:
        raise OutputValidationError("confidence does not match the top score")
    return obj


def build_decision_object(score_sales, threshold):
    """Model score -> decision object, with the threshold applied."""
    scores = {"sales": float(score_sales), "customer_service": float(1.0 - score_sales)}
    decision = "sales" if score_sales >= threshold else "customer_service"
    return {"decision": decision, "confidence": scores[decision], "scores": scores}


def route(obj, score_sales, threshold):
    """Validated decision object -> (final decision, routing flag, abstained?)."""
    if abs(score_sales - threshold) < ABSTAIN_BAND:
        return "uncertain", DECISION_TO_FLAG[FALLBACK_DECISION], True
    return obj["decision"], DECISION_TO_FLAG[obj["decision"]], False


def score_all_configs(messages):
    """Run every instruction variant over every message; return a tidy DataFrame."""
    from transformers import pipeline
    print(f"Loading {MODEL_NAME} from the local cache (HF_HUB_OFFLINE=1)...", flush=True)
    t0 = time.perf_counter()
    clf = pipeline("zero-shot-classification", model=MODEL_NAME, device=-1)
    load_s = time.perf_counter() - t0
    print(f"Loaded in {load_s:.1f}s.", flush=True)

    rows, t0 = [], time.perf_counter()
    for name, template, labels in CONFIGS:
        text2label = {v: k for k, v in labels.items()}
        print(f"  config {name} ...", flush=True)
        for mid, msg in messages:
            raw = clf(msg, candidate_labels=list(labels.values()),
                      hypothesis_template=template, multi_label=False)
            s = {text2label[l]: float(v) for l, v in zip(raw["labels"], raw["scores"])}
            rows.append({"config": name, "id": mid, "score_sales": s["sales"],
                         "score_customer_service": s["customer_service"]})
    total_s = time.perf_counter() - t0
    n = len(CONFIGS) * len(messages)
    print(f"{n} inferences in {total_s:.1f}s ({total_s / n:.2f} s/message-config)", flush=True)
    return pd.DataFrame(rows), load_s, total_s / n


def select_on_train(scores, train_df):
    """Pick (config, threshold) by accuracy on the training split only."""
    rows = []
    for name, _, _ in CONFIGS:
        s = scores[scores.config == name].merge(train_df[["id", "label"]], on="id")
        for t in THRESHOLD_GRID:
            pred = (s.score_sales >= t).astype(int)
            rows.append({"config": name, "threshold": float(t),
                         "train_accuracy": float((pred == s.label).mean())})
    sel = pd.DataFrame(rows)
    # Best train accuracy; ties -> threshold closest to 0.5 (least extreme), then config order.
    sel["tie_break"] = (sel.threshold - 0.5).abs()
    best = sel.sort_values(["train_accuracy", "tie_break"], ascending=[False, True]).iloc[0]
    return sel.drop(columns="tie_break"), best["config"], float(best.threshold), float(best.train_accuracy)


def plot_confidence(out, threshold, path):
    fig, ax = plt.subplots(figsize=(7.4, 4.6), facecolor=SURFACE)
    bins = np.linspace(0, 1, 21)
    ax.hist([out[out.label == SUPPORT].score_sales, out[out.label == SALES].score_sales],
            bins=bins, color=[COLOR_SUPPORT, COLOR_SALES],
            label=["true support (0)", "true sales (1)"], edgecolor=SURFACE, linewidth=2)
    ax.axvline(threshold, color=INK, lw=1.8)
    ax.axvspan(threshold - ABSTAIN_BAND, threshold + ABSTAIN_BAND, color=INK, alpha=0.08)
    ax.annotate(f"threshold {threshold:.2f}\n(shaded = abstention band)",
                (threshold, ax.get_ylim()[1] * 0.78), xytext=(8, 0), textcoords="offset points",
                fontsize=8, color=INK)
    ax.set_xlabel("Model score for the 'sales' hypothesis")
    ax.set_ylabel("Messages")
    ax.set_title("Zero-shot decision model: score distribution over all 60 messages", fontsize=11)
    ax.legend(frameon=True, facecolor=SURFACE, edgecolor=GRID, fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    ensure_dirs()
    df = pd.read_csv(DATASET_CSV).merge(pd.read_csv(GROUPS_CSV), on="id")
    train, test = df[df.split == "train"], df[df.split == "test"]

    scores, load_s, s_per_inf = score_all_configs(list(zip(df.id, df.message)))
    scores.to_csv(RESULTS_DIR / "zeroshot_scores_all_configs.csv", index=False)

    sel, best_config, threshold, train_acc = select_on_train(scores, train)
    sel.to_csv(RESULTS_DIR / "zeroshot_config_selection.csv", index=False)
    per_config_best = sel.groupby("config").train_accuracy.max().to_dict()
    print(f"\nBest train accuracy per config: {per_config_best}")
    print(f"Selected config {best_config} at threshold {threshold:.2f} (train acc {train_acc:.3f})")

    # ---- apply the selection to every message, through the wrapper ----
    chosen = scores[scores.config == best_config].set_index("id").score_sales
    rows, n_invalid = [], 0
    for _, r in df.iterrows():
        s_sales = float(chosen[r.id])
        obj = build_decision_object(s_sales, threshold)
        try:
            validate_decision_object(obj)
            valid = True
        except OutputValidationError as e:
            valid, n_invalid = False, n_invalid + 1
            print(f"  [invalid output] id={r.id}: {e}")
        decision, flag, abstained = route(obj, s_sales, threshold)
        rows.append({
            "id": r.id, "message": r.message, "label": r.label, "split": r.split,
            "group_id": r.group_id, "category": r.category,
            "score_sales": round(s_sales, 4), "score_customer_service": round(1 - s_sales, 4),
            "raw_decision": obj["decision"], "confidence": round(obj["confidence"], 4),
            "schema_valid": valid, "abstained": abstained,
            "final_decision": decision, "pred_flag": flag,
            "decision_json": json.dumps(obj, sort_keys=True),
        })

    out = pd.DataFrame(rows)
    out["correct"] = out.pred_flag == out.label
    out.to_csv(RESULTS_DIR / "zeroshot_predictions.csv", index=False)
    out_test = out[out.split == "test"]
    out_test[~out_test.correct].to_csv(RESULTS_DIR / "zeroshot_errors.csv", index=False)

    metrics_test = compute_metrics(out_test.label, out_test.pred_flag)
    metrics_all = compute_metrics(out.label, out.pred_flag)
    per_cat = out_test.groupby("category").correct.agg(["sum", "count"])

    plot_confusion_matrix(metrics_test["confusion_matrix"],
                          f"Zero-shot decision model (test, n={len(out_test)})",
                          RESULTS_DIR / "zeroshot_confusion_matrix.png")
    plot_confidence(out, threshold, RESULTS_DIR / "zeroshot_confidence.png")

    name, template, labels = next(c for c in CONFIGS if c[0] == best_config)
    save_json({
        "model": MODEL_NAME,
        "license": "MIT (model weights); MultiNLI corpus under its own terms",
        "setup": "local HuggingFace cache, HF_HUB_OFFLINE=1, CPU, no fine-tuning, no API key",
        "selection": "instruction variant and decision threshold chosen by accuracy on the 40 training messages only",
        "selected_config": name,
        "hypothesis_template": template,
        "candidate_labels": labels,
        "decision_threshold": threshold,
        "abstain_band": ABSTAIN_BAND,
        "fallback_decision": FALLBACK_DECISION,
        "train_accuracy_per_config": per_config_best,
        "train_accuracy": train_acc,
        "schema_invalid_outputs": n_invalid,
        "n_abstentions_total": int(out.abstained.sum()),
        "n_abstentions_test": int(out_test.abstained.sum()),
        "load_seconds": load_s,
        "seconds_per_message": s_per_inf,
        "test": metrics_test,
        "all_60": metrics_all,
        "test_accuracy_by_category": {c: f"{int(r['sum'])}/{int(r['count'])}" for c, r in per_cat.iterrows()},
        "mean_confidence_correct": float(out[out.correct].confidence.mean()),
        "mean_confidence_wrong": float(out[~out.correct].confidence.mean()) if (~out.correct).any() else None,
    }, RESULTS_DIR / "zeroshot_metrics.json")

    print(f"\nTest acc {metrics_test['accuracy']:.3f} | all-60 acc {metrics_all['accuracy']:.3f}")
    print("Confusion matrix (test) [[TN FP],[FN TP]]:", metrics_test["confusion_matrix"])
    print(f"Invalid outputs: {n_invalid} | abstentions (test): {int(out_test.abstained.sum())}")
    print("Per category (test):", {c: f"{int(r['sum'])}/{int(r['count'])}" for c, r in per_cat.iterrows()})
    print("Test errors:\n", out_test[~out_test.correct][["id", "message", "label", "pred_flag", "score_sales"]])


if __name__ == "__main__":
    main()
