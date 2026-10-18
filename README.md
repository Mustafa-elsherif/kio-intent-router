# KIO Intent Router

Intent router for **Agent KIO**, a customer-facing chatbot with two specialist
agents. One customer message in, one routing flag out:

```
0 = support   (problems, complaints, existing requests)
1 = sales     (new purchases, packages, quotes)
```

KIO uses the flag to call the selected agent. Routing picks the *workflow*;
retrieval (RAG) then finds information inside it - a wrong route selects the
wrong tools and the wrong knowledge base, and the answer cannot recover.

This repo builds the router **twice** and compares both on the same flag and the
same test messages:

1. **Traditional ML classifier** - numeric message features + an SVM / logistic
   regression, selected by cross-validation.
2. **Decision model** - a local open-source zero-shot model wrapped so it returns
   a *validated structured decision* (`customer_service` / `sales`), standing in
   for a commercial Decision API such as Jev by TypeSafe AI.

The specialist agents are simulated by functions that print their name - a full
chatbot or RAG pipeline is out of scope (university course, Project 1).

Full write-up: **[REPORT.md](REPORT.md)** | presentation plan:
**[PRESENTATION_OUTLINE.md](PRESENTATION_OUTLINE.md)** | the deck:
**[KIO_Routing_Presentation.pptx](KIO_Routing_Presentation.pptx)** (11 slides, speaker notes, 7:30)

---

## Results (20 held-out test messages)

| Metric | Traditional ML | Decision model |
|---|---|---|
| **Accuracy** | **0.600** | **0.950** |
| Macro F1 | 0.583 | 0.950 |
| Precision (sales) | 0.571 | 1.000 |
| Recall (sales) | 0.800 | 0.900 |
| Precision (support) | 0.667 | 0.909 |
| Recall (support) | **0.400** | **1.000** |
| plain / negation / mixed | 7/8, 3/8, 2/4 | 8/8, 7/8, 4/4 |
| Latency per message (CPU) | 0.06 ms | 546 ms |
| Labelled examples needed | 40, curve still rising | **0** |

Confusion matrices (rows = true, columns = predicted):

```
Traditional ML            Decision model
          pred 0  pred 1            pred 0  pred 1
true 0       4       6      true 0     10       0
true 1       2       8      true 1      1       9
```

The two routers agree on 65% of test messages. On the 7 where they disagree the
decision model is right 7 times and the classifier 0 - exact two-sided McNemar
**p = 0.0156**, significant at 0.05. Across all 60 messages the decision model
scores 0.850; the test half is the easier one, so both numbers are reported.

**Conclusion.** Deploy the decision model as KIO's router now: it is far more
accurate on exactly the messages that matter (negation and mixed intent), it
never sent a customer with a real problem to sales in our test, it needs no
labelled data, and its structured-output wrapper makes the flag safe to use in a
pipeline. The classifier is not broken - it is starved of data; its learning
curve is still climbing at 40 messages. Log every live decision with its
confidence and revisit the classifier once a few thousand real labelled messages
exist. Caveat: 20 test messages is a small sample, so the ranking is solid but
the exact numbers would move on a larger set.

---

## Dataset

`data/messages.csv` - 60 synthetic English messages (`id, message, label, split`),
30 support + 30 sales, written as 30 **intent groups** of one base message plus a
paraphrase. Categories: 24 plain, 20 negation, 16 mixed intent. Split 40 train /
20 test, class-balanced, split **at group level** so paraphrases never straddle
the split. Seed 42. `data/message_groups.csv` carries the group, category and
variant of every message.

Labeling guideline, agreed before any model was run: support = a problem or an
existing service; sales = a new purchase with no unresolved problem; **a message
mixing an unresolved problem with purchase interest is support.**

---

## Setup

Windows, PowerShell. Python 3.12 with `torch`, `transformers`, `scikit-learn`,
`pandas`, `matplotlib`, `seaborn`, `scipy`, `joblib`:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install torch transformers scikit-learn pandas matplotlib seaborn python-pptx
```

The decision model needs **`facebook/bart-large-mnli`** (~1.6 GB, MIT licensed) in
the local HuggingFace cache. It is downloaded once:

```powershell
.\venv\Scripts\python.exe -c "from transformers import pipeline; pipeline('zero-shot-classification', model='facebook/bart-large-mnli')"
```

After that every run is offline (`HF_HUB_OFFLINE=1`), CPU-only, with no API key.
The cache and the trained classifier are **not** committed - both are regenerated
by the commands below.

## Running

Run in this order; each script is standalone and writes its outputs to `results/`.

```powershell
$py = ".\venv\Scripts\python.exe"

& $py src\generate_dataset.py    # data/messages.csv, data/message_groups.csv   (seconds)
& $py src\train_classifier.py    # approach 1: 45-config CV search + test        (~1 min)
$env:HF_HUB_OFFLINE = "1"
& $py src\decision_model.py      # approach 2: 4 instruction variants x 60 msgs  (~3 min, CPU)
& $py src\compare.py             # comparison, McNemar, learning curve           (~1 min)
& $py src\agents.py              # end-to-end demo through both routers
& $py src\build_presentation.py  # KIO_Routing_Presentation.pptx                 (seconds)
```

`build_presentation.py` needs `python-pptx`. It re-checks every number that
appears on a slide against `results/*.json` and refuses to build if one drifted.

`src/agents.py` also routes a single message:

```powershell
& $py src\agents.py "my bill is wrong but I also want a faster plan"
```

## Repository layout

```
data/      messages.csv, message_groups.csv
src/       common.py              shared paths, metrics, plot styling
           features.py            the 11 hand-crafted numeric features
           generate_dataset.py    builds the 60-message dataset
           train_classifier.py    approach 1 - TF-IDF / hand features + SVM / logreg
           decision_model.py      approach 2 - zero-shot + structured-output wrapper
           compare.py             comparative analysis and figures
           agents.py              simulated specialist agents + routing demo
           build_presentation.py  renders the .pptx deck from the outline
results/   metrics (JSON), per-message predictions and errors (CSV), figures (PNG)
REPORT.md                method, features, settings, results, error analysis, recommendation
PRESENTATION_OUTLINE.md  8-minute, 3-student presentation plan
KIO_Routing_Presentation.pptx   the built deck: 11 slides, figures, speaker notes
```

### Key result files

| File | What it holds |
|---|---|
| `results/comparison_table.csv` | the head-to-head metrics table |
| `results/comparison_summary.json` | both approaches, McNemar test, agreement |
| `results/ml_cv_results.csv` | all 50 cross-validated ML configurations |
| `results/ml_feature_importance.csv` | permutation importance of the 11 features |
| `results/zeroshot_config_selection.csv` | instruction variant x threshold search |
| `results/*_errors.csv` | every misrouted message, per approach |
| `results/ml_decision_boundary_lexicon.png` | the classifier's boundary on readable axes |
| `results/ml_learning_curve.png` | how much labelled data the classifier still needs |

---

## Method notes

* **No leakage.** Model selection, instruction wording and the decision threshold
  were all chosen on the 40 training messages only. The 20 test messages were
  scored once, after selection.
* **Grouped cross-validation.** Paraphrase pairs are a CV group, so a model is
  never validated on the near-duplicate of a message it trained on.
* **Output validation.** Every decision-model answer is parsed into
  `{"decision", "confidence", "scores"}` and schema-checked; 0 of 60 failed. An
  abstention band of +/-0.05 around the threshold falls back to support, the safe
  default.
* **Every number here comes from a run that was actually executed**, including the
  ones that are unflattering to the trained classifier.

## Licenses and sources

* `facebook/bart-large-mnli` - **MIT** (model weights); MultiNLI corpus under its
  own permissive terms. See the model card.
* Yin, Hay & Roth (2019), *Benchmarking Zero-shot Text Classification* - the
  NLI-as-zero-shot-classification method.
* scikit-learn 1.9.1, transformers 5.20.1, torch 2.14.1+cpu, pandas 3.0.6.
* Project brief: *Project 1 - Routing for Agent KIO*.
