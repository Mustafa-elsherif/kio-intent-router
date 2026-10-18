"""
agents.py - the two specialist agents, simulated, and the KIO routing demo.

The brief asks for the specialist agents to be simulated by functions that print
their name: the real agents (LLM + search tools + RAG) are out of scope for
Project 1. This file is the only place where the routing flag is actually
*used*, which is what the router exists for.

Run it to see the end-to-end path for a handful of messages with both routers:

    python src/agents.py                 # built-in demo messages
    python src/agents.py "my bill is wrong"   # route one message

Outputs: results/routing_demo.txt (transcript of the demo run)
"""
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import joblib

from common import MODELS_DIR, RESULTS_DIR, SUPPORT, SALES, FLAG_NAMES, ensure_dirs

DEMO_MESSAGES = [
    "My internet has been down since yesterday, and I may want a faster plan later.",
    "How much does the business fiber plan cost?",
    "Nothing is wrong with my service, I just want a faster plan.",
    "I did not order this upgrade, so why am I being billed for it?",
]


def support_agent(message):
    """Specialist agent for flag 0 - problems, complaints, existing requests."""
    print(f"    -> SUPPORT AGENT handling: {message!r}")
    return "support_agent"


def sales_agent(message):
    """Specialist agent for flag 1 - new purchases, packages, quotes."""
    print(f"    -> SALES AGENT handling: {message!r}")
    return "sales_agent"


AGENTS = {SUPPORT: support_agent, SALES: sales_agent}


def dispatch(message, flag):
    """KIO uses the flag to call the selected agent."""
    return AGENTS[flag](message)


def ml_router():
    """Load the trained traditional-ML router. Returns message -> flag."""
    model = joblib.load(MODELS_DIR / "ml_router.joblib")
    return lambda m: int(model.predict([m])[0])


def decision_router():
    """Build the zero-shot decision router with the settings selected on train."""
    import json
    from decision_model import (CONFIGS, build_decision_object, validate_decision_object, route)
    meta = json.loads((RESULTS_DIR / "zeroshot_metrics.json").read_text(encoding="utf-8"))
    name, template, labels = next(c for c in CONFIGS if c[0] == meta["selected_config"])
    threshold = meta["decision_threshold"]
    from transformers import pipeline
    clf = pipeline("zero-shot-classification", model=meta["model"], device=-1)
    text2label = {v: k for k, v in labels.items()}

    def router(message):
        raw = clf(message, candidate_labels=list(labels.values()),
                  hypothesis_template=template, multi_label=False)
        s = {text2label[l]: float(v) for l, v in zip(raw["labels"], raw["scores"])}
        obj = validate_decision_object(build_decision_object(s["sales"], threshold))
        decision, flag, abstained = route(obj, s["sales"], threshold)
        return flag, decision, obj["confidence"], abstained

    return router


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def main():
    ensure_dirs()
    messages = sys.argv[1:] or DEMO_MESSAGES
    with open(RESULTS_DIR / "routing_demo.txt", "w", encoding="utf-8") as fh:
        out = Tee(sys.stdout, fh)
        print("Loading routers (zero-shot model loads from the local cache, ~10 s)...", file=out)
        ml = ml_router()
        dec = decision_router()
        for m in messages:
            print(f"\nKIO received: {m!r}", file=out)
            f_ml = ml(m)
            print(f"  [traditional ML]   flag {f_ml} = {FLAG_NAMES[f_ml]}", file=out)
            sys.stdout = out
            dispatch(m, f_ml)
            sys.stdout = out.streams[0]
            f_dm, decision, conf, abstained = dec(m)
            note = "  (abstained -> safe default)" if abstained else ""
            print(f"  [decision model]   decision {decision!r} conf {conf:.2f}{note} "
                  f"-> flag {f_dm} = {FLAG_NAMES[f_dm]}", file=out)
            sys.stdout = out
            dispatch(m, f_dm)
            sys.stdout = out.streams[0]
    print(f"\nTranscript written to {RESULTS_DIR / 'routing_demo.txt'}")


if __name__ == "__main__":
    main()
