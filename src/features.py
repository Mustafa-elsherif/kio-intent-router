"""
features.py - hand-crafted numeric features for the traditional ML router.

Why: the assignment asks for a classifier trained on *numeric message features*.
With only 40 training messages a pure bag-of-words representation has almost no
vocabulary overlap with unseen messages, so we also build a small, interpretable
numeric feature vector.

The two lexicons below were fixed a priori from the routing definition in the
project brief ("the support agent handles problems, complaints and existing
requests; the sales agent handles new purchases, service packages and quotes").
They were NOT tuned on the test split - the choice between feature sets is made
by cross-validation on the training split only.

Feature vector (11 numeric features per message):
  0 n_words                 length in words
  1 n_chars                 length in characters
  2 has_question_mark       1 if the message contains '?'
  3 support_hits            lexicon hits per 10 words (problem / existing service)
  4 sales_hits              lexicon hits per 10 words (buying / price / package)
  5 lexicon_margin          sales_hits - support_hits
  6 negation_hits           not / n't / never / no / nothing, per 10 words
  7 existing_hits           already / still / since / again / my current ...
  8 new_purchase_hits       buy / order / another / additional / sign up ...
  9 money_hits              price / cost / bill / charge / invoice / refund ...
 10 first_person_want       'i want' / 'i'd like' / 'i need' / 'we need' present
"""
import re

import numpy as np

SUPPORT_LEXICON = [
    "problem", "issue", "error", "broken", "break", "damaged", "crack", "fault", "fix",
    "repair", "complaint", "complain", "cancel", "refund", "overcharg", "wrong",
    "doesn't work", "not working", "won't", "stopped", "stops", "down", "disconnect",
    "drop", "fail", "crash", "late", "delay", "missing", "hasn't arrived", "nowhere",
    "can't log in", "login", "log in", "stuck", "promised", "support", "help",
]
SALES_LEXICON = [
    "price", "pricing", "cost", "how much", "quote", "buy", "purchase", "order",
    "subscribe", "sign up", "sign-up", "plan", "package", "bundle", "tier", "upgrade",
    "add-on", "add on", "discount", "deal", "offer", "recommend", "license", "interested",
    "enterprise", "premium", "business plan", "contract", "renew", "sales",
]
EXISTING_LEXICON = [
    "already", "still", "since", "again", "my current", "current plan", "last month",
    "last week", "yesterday", "my bill", "my account", "my order", "my router",
    "i have", "existing", "this month",
]
NEW_PURCHASE_LEXICON = [
    "buy", "purchase", "order", "another", "additional", "more", "second", "new ",
    "sign up", "subscribe", "add the", "add a", "add an", "switch up", "move from",
    "would you recommend", "i'd like to", "interested in",
]
MONEY_LEXICON = [
    "price", "cost", "bill", "invoice", "charge", "charged", "refund", "pay", "payment",
    "discount", "quote", "cheaper", "how much", "$", "euro",
]
NEGATION_LEXICON = ["not", "n't", "never", "no ", "nothing", "none"]

FEATURE_NAMES = [
    "n_words", "n_chars", "has_question_mark", "support_hits", "sales_hits",
    "lexicon_margin", "negation_hits", "existing_hits", "new_purchase_hits",
    "money_hits", "first_person_want",
]
WANT_RE = re.compile(r"\b(i want|i'd like|i would like|i need|we need|we want|i'm looking|"
                     r"i am looking|can i|could you|please)\b")


def _hits(text, lexicon):
    return sum(text.count(term) for term in lexicon)


def featurize_one(message):
    t = message.lower()
    words = re.findall(r"[a-z']+", t)
    n_words = max(len(words), 1)
    per10 = lambda n: 10.0 * n / n_words
    support = per10(_hits(t, SUPPORT_LEXICON))
    sales = per10(_hits(t, SALES_LEXICON))
    return np.array([
        len(words),
        len(message),
        1.0 if "?" in message else 0.0,
        support,
        sales,
        sales - support,
        per10(_hits(t, NEGATION_LEXICON)),
        per10(_hits(t, EXISTING_LEXICON)),
        per10(_hits(t, NEW_PURCHASE_LEXICON)),
        per10(_hits(t, MONEY_LEXICON)),
        1.0 if WANT_RE.search(t) else 0.0,
    ], dtype=float)


def featurize(messages):
    return np.vstack([featurize_one(m) for m in messages])


class HandcraftedFeatures:
    """Minimal sklearn-compatible transformer around featurize()."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return featurize(list(X))

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def get_params(self, deep=True):
        return {}

    def set_params(self, **kwargs):
        return self
