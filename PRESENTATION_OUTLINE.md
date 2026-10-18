# Presentation outline - Routing for Agent KIO

**8 minutes hard stop. 3 students, 3 chapters. Every number below is measured -
do not round differently on the slides.**

Suggested time split (leaves ~30 s of slack):

| | Chapter | Slides | Time |
|---|---|---|---|
| Student 1 | Problem + dataset | 3 | 2:15 |
| Student 2 | Decision model | 3 | 2:30 |
| Student 3 | Traditional ML + comparison + recommendation | 4 (+ a sources slide) | 2:45 |

---

## Student 1 - Problem statement and dataset (~2:15, 3 slides)

### Slide 1 - The routing problem (0:45)
* KIO is the chat entry point to two specialist agents: **support** (problems,
  complaints, existing requests) and **sales** (new purchases, packages, quotes).
* Our block is the **intent router**: message in, flag out. `0 = support`,
  `1 = sales`. KIO uses the flag to call the agent.
* Routing picks the *workflow*; retrieval (RAG) then finds information *inside*
  it. A wrong route means the wrong tools and the wrong knowledge base - the
  answer cannot recover.
* We built the router **twice** and compared the two on the same flag.
* *Show*: the architecture diagram from the brief, with our block highlighted.

### Slide 2 - The dataset (0:50)
* **60 synthetic English messages = 30 intent groups**, each group a base message
  plus a paraphrase. 30 support / 30 sales.
* Three categories on purpose: **plain** (24 messages), **negation** (20),
  **mixed intent** (16). Written so keywords mislead - support messages that say
  *upgrade* and *premium package*, sales messages that say *nothing is wrong*.
* **Labeling guideline, agreed before any model ran:** support = a problem or an
  existing service; sales = a new purchase with no unresolved problem;
  **mixed intent with an unresolved problem -> support.**
* **Split 40 train / 20 test**, balanced by class, split **at group level** so a
  paraphrase never lands on the other side of the split. Seed 42.

### Slide 3 - Sample rows + the conceptual difference (0:40)

| id | message | label | split | category |
|---|---|---|---|---|
| 1 | My internet connection keeps dropping every evening. | 0 | test | plain |
| 13 | I did not order this upgrade, so why am I being billed for it? | 0 | test | negation |
| 25 | Before I buy a second license, can you fix the error the first one keeps showing? | 0 | test | mixed |
| 33 | I'd like to buy three more licenses for my team. | 1 | test | plain |
| 43 | Nothing is wrong with my service, I just want a faster plan. | 1 | test | negation |
| 53 | My last issue was fixed quickly, thanks! Now I'd like to order the security add-on. | 1 | train | mixed |

* **How a decision model differs from a trained classifier** - one line each:
  * The classifier **learns from our labels**: it turns the message into numbers
    and fits a boundary. It knows only what the 40 training messages taught it.
  * The decision model **already understands English**: it is given the message
    plus *routing instructions in words* and returns a typed decision. **Zero
    training examples.** We tune wording, not weights.
* *Transition*: "Student 2 will show the decision model first."

---

## Student 2 - The decision model (~2:30, 3 slides)

### Slide 4 - What we used and why (0:50)
* A Decision API like **Jev (TypeSafe AI)** takes message + instructions and
  returns a typed decision. We had no paid API, so we used the open-source
  equivalent **with our own structured-output wrapper**.
* **`facebook/bart-large-mnli`** - BART-large, 407M parameters, fine-tuned on
  MultiNLI. **License: MIT** (weights); MultiNLI under its own permissive terms.
* **Setup:** local HuggingFace cache, `HF_HUB_OFFLINE=1`, **CPU, no API key, no
  network call, no training on our data.**
* **Method:** zero-shot via natural-language inference - the message is the
  premise, each label becomes a hypothesis, entailment scores are compared.

### Slide 5 - Inputs, instructions and validation (0:50)
* The **instructions** are the template + the two label descriptions:

```
"This customer message is about {}."
customer_service -> "a problem with a service the customer already has"
sales            -> "buying a new product, a price or an upgrade"
threshold = 0.41
```

* We wrote **4 instruction variants** and picked one **on the training split
  only**: 0.875 / 0.750 / 0.750 / 0.675 train accuracy. **Wording is worth 20
  points** - say this out loud, it is the key insight of the chapter.
* **Structured output + validation**: every answer is parsed into
  `{"decision", "confidence", "scores"}` and checked - decision in the enum,
  confidence a float in [0,1], scores over exactly the two labels summing to 1.
  **0 of 60 outputs failed validation.**
* **Abstention band +/-0.05** around the threshold -> returns `"uncertain"` and
  falls back to **support**, the safe default. 3 of 60 abstained, 0 in test.

### Slide 6 - Results and what they show (0:50)
* **Test accuracy 0.95**, macro F1 0.95, 20 test messages.
  Across all 60 messages **0.85** - say both, the test half is the easier one.
* Confusion matrix: `[[10, 0], [1, 9]]` - **it never sent a customer with a real
  problem to sales.**
* By category: plain 8/8, negation 7/8, **mixed 4/4**.
* **Show two real outputs** (copied from `results/zeroshot_predictions.csv`) - this
  is what the router actually returns, not a sketch:

```
id 33  "I'd like to buy three more licenses for my team."              -> flag 1
{"decision": "sales", "confidence": 0.906,
 "scores": {"customer_service": 0.094, "sales": 0.906}}

id 25  "Before I buy a second license, can you fix the error the       -> flag 0
        first one keeps showing?"
{"decision": "customer_service", "confidence": 0.704,
 "scores": {"customer_service": 0.704, "sales": 0.296}}
```

* **Successful routes to call out:** id 33 is the easy case; **id 25 is the win** -
  mixed intent, the words *buy* and *second license* are right there, and it still
  routes to support because the customer has an unresolved error. That is exactly
  the rule we wrote in the labeling guideline, obeyed by a model we never trained.
* **One error - id 43, "Nothing is wrong with my service, I just want a faster
  plan."** It scored 0.07 for sales: confidently wrong, outside the abstention
  band. Lesson: *validation guarantees the shape of the answer, never its truth.*
* **Cost:** 546 ms per message on CPU, ~1.6 GB in memory.
* *Show*: `zeroshot_confusion_matrix.png`, `zeroshot_confidence.png`. If time
  allows, run `python src/agents.py` live - it prints the flag and the agent name.
* *Transition*: "Now the classical approach, and the head-to-head."

---

## Student 3 - Traditional ML, comparison, recommendation (~2:45, 4 slides)

### Slide 7 - The classifier: features and training (0:45)
* **Numeric features.** Two options were tried: **TF-IDF** word n-grams, and
  **11 hand-crafted features** - length, `?` present, support-lexicon hits per 10
  words, sales-lexicon hits, their margin, negation hits, existing-service hits,
  new-purchase hits, money words, first-person request. Lexicons fixed **a priori**
  from the routing definition.
* **Model search:** linear SVM, degree-2 polynomial-kernel SVM, and logistic
  regression on degree-2 polynomial features - **45 configurations**, selected by
  **5-fold grouped cross-validation on the 40 training messages only** (paraphrase
  pairs kept in the same fold).
* **Selected: 11 hand-crafted features + SVM, degree-2 polynomial kernel, C = 1.**
  CV accuracy 0.825. TF-IDF came last (0.725) - with 40 messages there is almost
  no vocabulary overlap with unseen text.

### Slide 8 - The boundary and the results (0:45)
* *Show*: **`ml_decision_boundary_lexicon.png`** - the boundary on two readable
  axes, support-lexicon hits vs sales-lexicon hits. (`ml_decision_boundary.png`
  shows the same for a PCA projection of the full feature space.)
* **Train accuracy 0.975 -> test accuracy 0.600.** That gap is overfitting: a
  degree-2 kernel separates 40 points easily, and CV on 40 points is too noisy to
  catch it.
* **Show its outputs**: `ml_test_scores.png` - the signed decision score of all 20
  test messages, one bar each, coloured by true label, with the misrouted ones
  marked. The per-message table behind it is `results/ml_predictions.csv`.
  Point at the bars sitting just past zero: ids 12 (score 0.02) and 17 (0.19) are
  support messages pushed over the line by a hair.
* **Permutation importance:** the strongest single feature is the **question
  mark** (0.080 accuracy drop); **negation is almost unused** (0.007). That one
  fact explains the errors on the next slide.
* Latency **well under 0.1 ms/message** (median of 51 repeats; 0.04-0.06 ms across runs) - about **four orders of magnitude faster** than the decision model.

### Slide 9 - Head to head (0:45)

| Metric (test, n=20) | Traditional ML | Decision model |
|---|---|---|
| Accuracy | **0.600** | **0.950** |
| Macro F1 | 0.583 | 0.950 |
| Recall (support) | **0.400** | **1.000** |
| Recall (sales) | 0.800 | 0.900 |
| plain / negation / mixed | 7/8, 3/8, 2/4 | 8/8, 7/8, 4/4 |
| Latency per message | < 0.1 ms | 546 ms |
| Labelled data needed | 40 and still rising | **0** |

* The routers **agree on 65%** of test messages. Where they disagree, the
  decision model is right **7 times, the classifier 0**. Exact McNemar
  **p = 0.0156 - significant**, even at n = 20.
* **The error pattern:** the classifier breaks on **negation and mixed intent**.
  *"I did not order this upgrade, so why am I being billed for it?"* -> it sees
  *upgrade*, *order*, `?` and routes to **sales**. The decision model reads the
  sentence and routes to support.
* **Fair to the classifier:** the top six CV configurations score 0.60-0.75 on
  test, and the learning curve (`ml_learning_curve.png`) is **still rising at 40
  messages**. It is not broken - it is **starved of data**.

### Slide 10 - Recommendation for KIO (0:30)
* **Deploy the decision model now.** Better where it matters (0.95 vs 0.60,
  significant), **never misroutes a real problem** in our test, **needs no
  labelled data**, and the wrapper makes the flag safe to use: typed output,
  confidence exposed, abstain -> support.
* **What we accept:** 546 ms per message and 1.6 GB of memory; MIT-licensed and
  fully offline, so no per-call fee and no customer data leaves the machine.
  Instruction wording must be version-controlled - it is worth 20 points.
* **The path forward:** log every decision with its confidence, let agents flag
  misroutes, and after a few thousand real labelled messages retrain the
  classifier on that log and re-run this exact comparison. The cheap, fast model
  becomes attractive again **once the data exists**.
* **Caveat we state ourselves:** 20 test messages is small - the same decision
  model scores 0.85 over all 60. The ranking is solid; the exact numbers would
  move on a larger set.

### Slide 11 - Sources (on screen while Student 3 closes and through Q&A; no extra speaking time)

* Project 1 brief, *Routing for Agent KIO* - the task, the flag convention and
  the dataset specification.
* `facebook/bart-large-mnli` model card, HuggingFace - **MIT license** (weights);
  MultiNLI corpus under its own permissive terms.
* Yin, Hay & Roth (2019), *Benchmarking Zero-shot Text Classification: Datasets,
  Evaluation and Entailment Approach* - the NLI-as-zero-shot-classifier method we
  use.
* Williams, Nangia & Bowman (2018), *A Broad-Coverage Challenge Corpus for
  Sentence Understanding through Inference* - MultiNLI, the corpus the model was
  fine-tuned on.
* Dietterich (1998), *Approximate statistical tests for comparing supervised
  classification learning algorithms* - McNemar's test for paired comparison.
* Software: scikit-learn 1.9.1, transformers 5.20.1, torch 2.14.1+cpu,
  pandas 3.0.6, matplotlib 3.11.2, seaborn 0.13.2, scipy 1.18.1, Python 3.12.10.
* All code, data and saved results: the project repository (`src/`, `data/`,
  `results/`, `REPORT.md`).
* Dataset: 60 messages written by our team for this project, synthetic, no
  customer data.

---

## Rehearsal checklist
* All three students speak; hand-offs are scripted (end of slides 3 and 6).
* Figures open **before** the talk: `ml_decision_boundary_lexicon.png`,
  `ml_learning_curve.png`, `comparison_accuracy.png`, `comparison_by_category.png`,
  `zeroshot_confusion_matrix.png`, `zeroshot_confidence.png`.
* If the live demo is used, **load the model before the presentation starts**
  (~9 s) - otherwise show `results/routing_demo.txt`.
* **Hard stop at 8:00.** If running late, drop slide 8's permutation-importance
  bullet and the caveat on slide 10 - never the recommendation.
* **Slide 11 (sources) must be on screen** before Student 3 finishes - the brief
  requires work, results *and* sources to be presented.
