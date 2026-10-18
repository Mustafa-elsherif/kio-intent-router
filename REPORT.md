# Project 1 - Intent Routing for Agent KIO

**Task.** KIO is a customer-facing chatbot with two specialist agents. Our block is
the *intent router*: message in, routing flag out, `0 = support`, `1 = sales`.
KIO uses the flag to call the selected agent. We built the router twice - once as
a **trained ML classifier**, once as a **decision model** - and compared both on
the same flag and the same test messages.

All numbers in this report come from runs executed on this machine. Every script
saves its own outputs to `results/`, so nothing here is recomputed by hand.

---

## 1. Dataset

### 1.1 Generation method

60 synthetic English customer messages, written as **30 intent groups**. Each
group is one base message plus one paraphrase of it, so paraphrases are
near-duplicates by construction. 15 groups are support, 15 are sales.

Each group carries a category:

| Category | Groups | Messages | What it tests |
|---|---|---|---|
| `plain` | 12 | 24 | the straightforward case |
| `negation` | 10 | 20 | "I did **not** order this", "**Nothing** is wrong" |
| `mixed` | 8 | 16 | a problem and a purchase in one message |

The messages were deliberately written so that **keywords alone are misleading**:
support messages mention *upgrade*, *premium package*, *sales rep*; sales messages
contain *nothing is wrong*, *no problems*, *doesn't*.

### 1.2 Labeling guideline (agreed before any model was run)

* **0 - support**: a problem, fault, complaint, billing error, missing delivery,
  refund, account-access issue, or a request about an existing service
  (including cancellation).
* **1 - sales**: a new purchase, upgrade, add-on, price, quote or plan
  recommendation, with **no unresolved existing problem**.
* **Priority rule for mixed intent**: if a message mixes an *unresolved* existing
  problem with purchase interest, it is **0 - support**. A problem that the
  message states is already resolved does not count.

### 1.3 Split

40 training / 20 test, class-balanced, split **at group level** with seed 42, so a
paraphrase and its base message are always in the same split - a model is never
tested on the near-duplicate of something it trained on. Categories are
stratified too.

| Split | support | sales | plain | negation | mixed |
|---|---|---|---|---|---|
| train | 20 | 20 | 16 | 12 | 12 |
| test | 10 | 10 | 8 | 8 | 4 |

Test groups: `L02, L06, L07, L11, L15, S01, S06, S07, S09, S13`.

### 1.4 Sample rows

| id | message | label | split | category |
|---|---|---|---|---|
| 1 | My internet connection keeps dropping every evening. | 0 | test | plain |
| 13 | I did not order this upgrade, so why am I being billed for it? | 0 | test | negation |
| 25 | Before I buy a second license, can you fix the error the first one keeps showing? | 0 | test | mixed |
| 33 | I'd like to buy three more licenses for my team. | 1 | test | plain |
| 43 | Nothing is wrong with my service, I just want a faster plan. | 1 | test | negation |
| 53 | My last issue was fixed quickly, thanks! Now I'd like to order the security add-on. | 1 | train | mixed |

Files: `data/messages.csv` (`id, message, label, split`), `data/message_groups.csv`
(`id, group_id, category, variant`), summary in `results/dataset_summary.json`.

---

## 2. Approach 1 - Traditional ML classifier

### 2.1 Numeric features

Two representations were built and both were offered to the model search:

* **TF-IDF** - weights of word n-grams, lower-cased, sublinear tf, L2-normalised.
  Stop words are **kept**, because *not / no / never* carry the negation cases.
* **HAND** - 11 interpretable numeric features (`src/features.py`): message length
  in words and characters, presence of `?`, support-lexicon hits per 10 words,
  sales-lexicon hits per 10 words, their margin, negation hits, existing-service
  hits, new-purchase hits, money-word hits, and a first-person-request flag.
  The two lexicons were fixed **a priori** from the routing definition in the
  brief and never tuned on the test split.
* **UNION** - the concatenation of both.

### 2.2 Model selection

Model families allowed by the brief were all searched: **linear SVM**,
**degree-2 polynomial-kernel SVM**, and **logistic regression on degree-2
polynomial features** (the polynomial *expansion* is only run on the 11
hand-crafted features - expanding thousands of TF-IDF columns to their pairwise
products is intractable here; the polynomial *kernel* covers the non-linear case
for the sparse representations).

Selection was by **5-fold StratifiedGroupKFold cross-validation on the 40
training messages only**, with paraphrase pairs as groups. 50 configurations were
scored (`results/ml_cv_results.csv`). The test split was touched exactly once,
after selection.

Best CV accuracy per feature set:

| Feature set | best CV accuracy |
|---|---|
| **hand** | **0.825** |
| union | 0.800 |
| tfidf (1,1) | 0.725 |
| tfidf (1,2) | 0.725 |

**Selected: `hand` features + SVM with a degree-2 polynomial kernel, C = 1**
(`coef0 = 1.0`, `gamma = "scale"`, features standardised), CV accuracy
0.825 +/- 0.100, 11 features.

Pure TF-IDF scored worst, and that is the expected result at this scale: with 40
training messages the unseen test messages share almost no vocabulary with the
training ones, so bag-of-words has nothing to generalise from.

### 2.3 What the model learned

Permutation importance on the training split (accuracy drop when one feature is
shuffled, 50 repeats - `results/ml_feature_importance.csv`):

| feature | accuracy drop |
|---|---|
| has_question_mark | 0.080 |
| new_purchase_hits | 0.058 |
| support_hits | 0.054 |
| lexicon_margin | 0.043 |
| existing_hits | 0.041 |
| first_person_want | 0.040 |
| negation_hits | 0.007 |
| n_words | 0.001 |

The classifier leans most on a **surface cue - the question mark** - and barely
uses the negation feature at all. That single line explains most of its errors
in section 4.

### 2.4 Results

Trained on 40, evaluated once on the 20 test messages:

* train accuracy **0.975**, test accuracy **0.600**, macro F1 **0.583**
* prediction latency **0.23 ms/message** on CPU

Figures: `ml_decision_boundary.png` (boundary in a 2D PCA projection of the
selected feature space; the 2D model itself scores 0.825 train / 0.650 test),
`ml_decision_boundary_lexicon.png` (boundary on the two raw lexicon features -
readable axes, 0.825 train / 0.750 test), `ml_test_scores.png`,
`ml_confusion_matrix.png`.

The gap between 0.975 train and 0.600 test is plain **overfitting**: the degree-2
kernel can separate 40 points almost perfectly, and CV on 40 points is too noisy
to catch it (fold accuracies ranged 0.75-1.00).

---

## 3. Approach 2 - Decision model (zero-shot, structured output)

A commercial Decision API such as **Jev by TypeSafe AI** takes a message plus
routing instructions and returns a typed decision. We had no paid API, so we used
a **local open-source model with our own structured-output wrapper**, which keeps
the property that matters for a pipeline: the output is a validated object, not
free text.

| | |
|---|---|
| Model | `facebook/bart-large-mnli` - BART-large (407M params) fine-tuned on MultiNLI |
| License | **MIT** for the model weights; the MultiNLI training corpus is distributed under its own permissive terms (see the model card) |
| Setup | pre-downloaded into the local HuggingFace cache, run with `HF_HUB_OFFLINE=1`, **CPU only, no API key, no network call, no training on our data** |
| Method | zero-shot classification by natural-language inference: the message is the premise, each candidate label is verbalised into a hypothesis, entailment scores are softmaxed against each other |

### 3.1 Inputs and routing instructions

The "instructions" are the hypothesis template plus the two label verbalisations.
Four variants were written and **selected on the 40 training messages only**:

| Variant | Train accuracy |
|---|---|
| `A_short_team` - "handled by the {customer service / sales} team" | 0.750 |
| **`B_topic` - "This customer message is about {...}."** | **0.875** |
| `C_long_routing` - long "routed to the ... team, because ..." | 0.750 |
| `D_intent` - "The customer's intent is {...}." | 0.675 |

**Selected (`B_topic`):**

```
hypothesis template : "This customer message is about {}."
customer_service    : "a problem with a service the customer already has"
sales               : "buying a new product, a price or an upgrade"
decision threshold  : 0.41   (also selected on the training split)
```

The wording matters a lot: the worst variant is 20 points below the best. This is
the decision-model equivalent of hyperparameter tuning, and it is **free of
labelled training in the ML sense** - it needs a handful of examples to sanity-check
the instructions, not a training set.

### 3.2 Output validation

The wrapper (`src/decision_model.py`) turns the raw scores into

```json
{"decision": "sales", "confidence": 0.95,
 "scores": {"customer_service": 0.05, "sales": 0.95}}
```

and rejects anything that fails the schema: `decision` must be in the enum
`{customer_service, sales}`, `confidence` must be a float in `[0, 1]`, `scores`
must cover exactly the two allowed labels, sum to 1.0 (+/-1e-3), and agree with
`confidence`. A failure raises `OutputValidationError`.

**0 of 60 outputs failed validation.**

On top of the schema there is an **abstention band** of +/-0.05 around the
threshold: inside it the wrapper returns `"uncertain"` and the router falls back
to `customer_service` (flag 0), so an unsure sales guess never strands a customer
who has a real problem. 3 of 60 messages abstained (0 of the 20 test messages).

### 3.3 Results

* test accuracy **0.950**, macro F1 **0.950**
* across all 60 messages **0.850** (the test split happens to be the easier half -
  we report both so the headline number is not oversold)
* latency **546 ms/message** on CPU; model load ~9 s

Figures: `zeroshot_confusion_matrix.png`, `zeroshot_confidence.png`.

---

## 4. Comparison

Both routers map to the same flag and were scored on the same 20 test messages.

| Metric (test, n=20) | Traditional ML | Decision model |
|---|---|---|
| **Accuracy** | **0.600** | **0.950** |
| Precision (sales) | 0.571 | 1.000 |
| Recall (sales) | 0.800 | 0.900 |
| F1 (sales) | 0.667 | 0.947 |
| Precision (support) | 0.667 | 0.909 |
| Recall (support) | 0.400 | 1.000 |
| F1 (support) | 0.500 | 0.952 |
| **Macro F1** | **0.583** | **0.950** |

Confusion matrices (rows = true, columns = predicted):

```
Traditional ML            Decision model
          pred 0  pred 1            pred 0  pred 1
true 0       4       6      true 0     10       0
true 1       2       8      true 1      1       9
```

The ML router sends **6 of 10 customers with a real problem to the sales team**.
In the business scenario that is the expensive error.

**Accuracy by category (test):**

| Category | Traditional ML | Decision model |
|---|---|---|
| plain (n=8) | 7/8 | 8/8 |
| negation (n=8) | 3/8 | 7/8 |
| mixed (n=4) | 2/4 | 4/4 |

**Paired comparison.** The two routers agree on 65% of the test messages. On the
7 messages where they differ, the decision model is right **7 times and the ML
router 0 times**. Exact two-sided McNemar: **p = 0.0156** - significant at 0.05,
even on 20 messages. 12 messages both get right, 1 both get wrong.

### 4.1 Error analysis

Every ML error (`results/ml_errors.csv`, `results/error_overlap.csv`):

| id | true | ML | DM | Message | Why ML failed |
|---|---|---|---|---|---|
| 12 | 0 | **1** | 0 | "It's been fourteen days and the package I ordered is nowhere to be seen." | *ordered* fires the purchase lexicon; the complaint is in the sentence structure, not the words |
| 13 | 0 | **1** | 0 | "I did not order this upgrade, so why am I being billed for it?" | *upgrade*, *order*, and a `?` - the negation feature barely counts |
| 14 | 0 | **1** | 0 | "I never asked for the premium package, but it is on my invoice." | same: *premium package* outweighs *never asked* |
| 17 | 0 | **1** | 0 | "The app doesn't open on my phone anymore." | short message, no support keyword in the lexicon, no `?` |
| 25 | 0 | **1** | 0 | "Before I buy a second license, can you fix the error the first one keeps showing?" | mixed intent: *buy*, *second*, `?` dominate *fix the error* |
| 26 | 0 | **1** | 0 | "The license I already have keeps throwing an error; I'd like that sorted before purchasing another." | mixed intent, same cause |
| 43 | 1 | **0** | **0** | "Nothing is wrong with my service, I just want a faster plan." | *nothing is wrong* + *service* read as a complaint - **both routers fail this one** |
| 44 | 1 | **0** | 1 | "No problems at all with my connection; I simply want to upgrade the speed." | same shape; the decision model gets it right |

The single decision-model error is id 43, the hardest negation in the set: a
sales message built almost entirely out of support vocabulary. It scored 0.07 for
sales - confidently wrong, well outside the abstention band, so the wrapper did
not catch it. That is the honest limitation of the approach: validation checks
the *shape* of the answer, never its correctness.

Pattern: **ML fails on negation and mixed intent (5/8 negation, 2/4 mixed wrong)
because its features are a bag of surface cues. The decision model reads the
sentence.**

### 4.2 Honesty notes on the ML number

Two checks were run so the 0.600 is not misread:

* **Post-hoc sensitivity** (`results/ml_posthoc_sensitivity.csv`): the top six CV
  configurations score between **0.60 and 0.75** on the test set. They were *not*
  used to select the model - reporting them after the fact shows how unstable the
  choice is at this dataset size. Even the luckiest of them stays 20 points below
  the decision model.
* **Learning curve** (`results/ml_learning_curve.png`): grouped-CV accuracy of the
  selected configuration rises from 0.69 at 12 training messages to **0.83 at 40**
  and is still climbing. The trained approach is not broken - it is **starved of
  data**. The decision model reaches 0.95 on test with **zero labelled training
  examples**.

---

## 5. Recommendation for KIO

**Deploy the decision model as the router now; keep the trained classifier as the
plan for later.**

Reasons, in order of weight:

1. **It is better where it matters.** 0.95 vs 0.60 accuracy, and the difference is
   statistically significant (McNemar p = 0.0156). It never routes a customer with
   a real problem to sales in our test (support recall 1.00 vs 0.40).
2. **It needs no labelled data.** The trained classifier would need several
   hundred labelled messages to close the gap - our learning curve is still rising
   at 40. That is weeks of annotation before KIO routes anything.
3. **It handles the hard cases.** Negation and mixed intent are exactly the real
   customer messages that matter, and those are where the classifier collapses
   (3/8 and 2/4).
4. **The structured-output wrapper makes it safe to use in a pipeline.** The flag
   is validated, confidence is exposed, and the abstention band gives a safe
   default: when unsure, send the customer to support.

Costs we accept, and how to manage them:

* **Latency**: 546 ms vs 0.23 ms per message - roughly 2 400x slower. Acceptable
  for a chat turn; if it is not, cache repeated messages or batch on GPU.
* **Size**: a 1.6 GB model in memory versus a few kilobytes. It is MIT-licensed
  and runs offline, so there is no per-call fee and no customer data leaving the
  machine.
* **Prompt sensitivity**: label wording swung training accuracy by 20 points.
  Treat the instruction text as a configuration artefact: version it, and re-run
  the 40-message check before changing it.
* **Confidently wrong on id 43.** Validation guarantees shape, not truth.

**Suggested path.** Route with the decision model from day one, log every decision
with its confidence, and have agents flag misroutes. After a few thousand real
labelled messages, retrain the classifier on that log and re-run this exact
comparison - the cheap, fast classifier becomes attractive again once the data
exists. Until then, a hybrid is also available for free: use the classifier when
both routers agree (65% of messages here, 12/13 of which are correct) and the
decision model otherwise.

**Threat to validity.** 20 test messages is a small sample; the 0.95 carries a
wide confidence interval, and the same model scores 0.85 over all 60. The ranking
is robust (the paired test is significant and no reasonable ML configuration gets
near 0.95), but the exact numbers would move on a larger set.

---

## 6. Reproducing this

```powershell
$py = "D:\year3\kio_project\venv\Scripts\python.exe"
& $py src\generate_dataset.py     # data/messages.csv  (seconds)
& $py src\train_classifier.py     # approach 1          (~1 min)
$env:HF_HUB_OFFLINE = "1"
& $py src\decision_model.py       # approach 2          (~3 min, CPU)
& $py src\compare.py              # comparison          (~1 min)
& $py src\agents.py               # end-to-end demo
```

`src/agents.py` simulates the two specialist agents with functions that print
their name, and shows KIO dispatching on the flag from both routers
(`results/routing_demo.txt`). The real agents - LLM, search tools, RAG - are out
of scope for Project 1.

### Sources

* Project 1 brief, *Routing for Agent KIO*.
* `facebook/bart-large-mnli` model card, HuggingFace (MIT license).
* Yin, Hay & Roth (2019), *Benchmarking Zero-shot Text Classification: Datasets,
  Evaluation and Entailment Approach* - the NLI-as-zero-shot-classifier method.
* Williams, Nangia & Bowman (2018), *A Broad-Coverage Challenge Corpus for
  Sentence Understanding through Inference* - MultiNLI, the fine-tuning corpus.
* Dietterich (1998), *Approximate statistical tests for comparing supervised
  classification learning algorithms* - McNemar's test for paired comparison.
* scikit-learn 1.9.1 (`SVC`, `StratifiedGroupKFold`, `permutation_importance`),
  transformers 5.20.1, torch 2.14.1+cpu, pandas 3.0.6, matplotlib 3.11.2,
  seaborn 0.13.2, scipy 1.18.1, Python 3.12.10.
