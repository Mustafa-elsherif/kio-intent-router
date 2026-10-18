"""
generate_dataset.py - builds the 60-message synthetic KIO routing dataset.

What it does
  * Holds 30 hand-written "intent groups" (15 support, 15 sales). Each group is
    one base message plus one paraphrase of it, so the two are near-duplicates.
  * Every group has a category: plain, negation, or mixed intent.
  * Labels were fixed by the labeling guideline below BEFORE any model was run.
  * Splits 40 train / 20 test at GROUP level (paraphrases never straddle the
    split), balanced by class and stratified by category, with a fixed seed.

Outputs
  data/messages.csv        id, message, label, split   (label: 0 support, 1 sales)
  data/message_groups.csv  id, group_id, category, variant
  results/dataset_summary.json

Labeling guideline (agreed before testing)
  0 support : a problem, fault, complaint, billing error, missing delivery or
              refund, account-access issue, or a request about an existing
              service such as cancellation.
  1 sales   : a new purchase, upgrade, add-on, price, quote or plan
              recommendation, with NO unresolved existing problem.
  Priority  : if a message mixes an unresolved existing problem with purchase
              interest, it is labeled 0 support. A problem that is explicitly
              already resolved does not count.

Generation method: the messages were written by an AI assistant (Claude) acting
as the dataset author under the guideline above; groups were designed so that
keywords alone are misleading (support messages that mention "upgrade",
"package", "sales rep"; sales messages that mention "problem", "not", "no").
"""
import random

import pandas as pd

from common import DATASET_CSV, GROUPS_CSV, RESULTS_DIR, SEED, ensure_dirs, save_json

# (group_id, label, category, [base, paraphrase])
GROUPS = [
    # ---------------- support (0) ----------------
    ("S01", 0, "plain", ["My internet connection keeps dropping every evening.",
                         "Every night my Wi-Fi disconnects over and over again."]),
    ("S02", 0, "plain", ["I was charged twice for my subscription this month.",
                         "There are two identical charges for my plan on this month's bill."]),
    ("S03", 0, "plain", ["I can't log in to my account since the last update.",
                         "Since your latest update my login stops working."]),
    ("S04", 0, "plain", ["The router you sent me arrived damaged.",
                         "My new router came in the box with a cracked case."]),
    ("S05", 0, "plain", ["I want to cancel my subscription.",
                         "Please close my account and stop the monthly payments."]),
    ("S06", 0, "plain", ["My order still hasn't arrived after two weeks.",
                         "It's been fourteen days and the package I ordered is nowhere to be seen."]),
    ("S07", 0, "negation", ["I did not order this upgrade, so why am I being billed for it?",
                            "I never asked for the premium package, but it is on my invoice."]),
    ("S08", 0, "negation", ["I'm not looking to buy anything, I just need my router fixed.",
                            "No new products please, I only want my broken modem repaired."]),
    ("S09", 0, "negation", ["The app doesn't open on my phone anymore.",
                            "Your app won't start on my phone no matter what I try."]),
    ("S10", 0, "negation", ["I haven't received the refund you promised last month.",
                            "The refund I was promised still hasn't shown up in my account."]),
    ("S11", 0, "negation", ["The discount I was promised is not applied to my bill.",
                            "My bill doesn't show the discount your agent promised me."]),
    ("S12", 0, "mixed", ["My internet has been down since yesterday; also, do you have faster plans?",
                         "Nothing has worked on my line since yesterday, and I might want to upgrade later."]),
    ("S13", 0, "mixed", ["Before I buy a second license, can you fix the error the first one keeps showing?",
                         "The license I already have keeps throwing an error; I'd like that sorted before purchasing another."]),
    ("S14", 0, "mixed", ["I'd love to upgrade to the premium plan, but my current one has been overcharging me.",
                         "I'm thinking of buying premium, but first explain why my current bill is too high."]),
    ("S15", 0, "mixed", ["Your sales rep sold me a package that doesn't include the channels I was promised.",
                         "The bundle I bought from your sales team is missing the channels they promised."]),
    # ---------------- sales (1) ----------------
    ("L01", 1, "plain", ["How much does the business fiber plan cost?",
                         "What's the price of your fiber package for businesses?"]),
    ("L02", 1, "plain", ["I'd like to buy three more licenses for my team.",
                         "Can I purchase three additional user licenses for my staff?"]),
    ("L03", 1, "plain", ["Can you send me a quote for 50 employee accounts?",
                         "Please prepare a price quote for fifty staff accounts."]),
    ("L04", 1, "plain", ["Which package would you recommend for a small office?",
                         "What plan do you suggest for an office with ten people?"]),
    ("L05", 1, "plain", ["Do you offer a discount if I pay yearly?",
                         "Is there a cheaper price for an annual subscription?"]),
    ("L06", 1, "plain", ["I want to sign up for your mobile plan.",
                         "How do I subscribe to the new mobile package?"]),
    ("L07", 1, "negation", ["Nothing is wrong with my service, I just want a faster plan.",
                            "No problems at all with my connection; I simply want to upgrade the speed."]),
    ("L08", 1, "negation", ["I don't have an account yet, but I'm interested in your TV bundle.",
                            "I'm not a customer yet - can you tell me about the TV package?"]),
    ("L09", 1, "negation", ["I don't need technical help, I need pricing for the enterprise package.",
                            "This is not a support question: what does the enterprise tier cost?"]),
    ("L10", 1, "negation", ["I'm not unhappy with anything; I just want to add a second phone line.",
                            "No complaints here, I'd only like an extra line on my plan."]),
    ("L11", 1, "negation", ["Doesn't the premium plan include international calls? If it does, I'll buy it.",
                            "If the premium package has international calling, I'm ready to purchase it."]),
    ("L12", 1, "mixed", ["My last issue was fixed quickly, thanks! Now I'd like to order the security add-on.",
                         "Thanks for solving my problem last week; next I want to buy the security add-on."]),
    ("L13", 1, "mixed", ["I'm happy with my current plan and want to add the streaming package.",
                         "My current subscription works great; please add the streaming bundle to it."]),
    ("L14", 1, "mixed", ["My contract ends next month - what upgrade offers do you have for loyal customers?",
                         "Since my contract is ending soon, which renewal deals can you offer me?"]),
    ("L15", 1, "mixed", ["Our team is growing, so we need to move from the basic to the business package.",
                         "We're hiring more people and want to switch up to the business plan."]),
]

# Test groups per class and category (5 groups = 10 messages per class).
TEST_GROUPS_PER_CATEGORY = {"plain": 2, "negation": 2, "mixed": 1}


def choose_test_groups(rng):
    test = set()
    for label in (0, 1):
        for cat, k in TEST_GROUPS_PER_CATEGORY.items():
            ids = sorted(g for g, lab, c, _ in GROUPS if lab == label and c == cat)
            test.update(rng.sample(ids, k))
    return test


def main():
    ensure_dirs()
    rng = random.Random(SEED)
    test_groups = choose_test_groups(rng)

    rows, meta = [], []
    msg_id = 1
    for gid, label, cat, texts in GROUPS:
        split = "test" if gid in test_groups else "train"
        for variant, text in zip(("base", "paraphrase"), texts):
            rows.append({"id": msg_id, "message": text, "label": label, "split": split})
            meta.append({"id": msg_id, "group_id": gid, "category": cat, "variant": variant})
            msg_id += 1

    df = pd.DataFrame(rows)
    meta_df = pd.DataFrame(meta)

    # Sanity checks on the deliverable's constraints.
    assert len(df) == 60 and df.message.is_unique
    assert (df.label == 0).sum() == 30 and (df.label == 1).sum() == 30
    counts = df.groupby(["split", "label"]).size().to_dict()
    assert counts == {("test", 0): 10, ("test", 1): 10, ("train", 0): 20, ("train", 1): 20}, counts
    merged = df.merge(meta_df, on="id")
    assert (merged.groupby("group_id").split.nunique() == 1).all()

    df.to_csv(DATASET_CSV, index=False)
    meta_df.to_csv(GROUPS_CSV, index=False)

    summary = {
        "n_messages": len(df),
        "n_groups": len(GROUPS),
        "seed": SEED,
        "counts_split_label": {f"{s}_{'support' if l == 0 else 'sales'}": int(n)
                               for (s, l), n in counts.items()},
        "counts_split_category": {f"{s}_{c}": int(n) for (s, c), n in
                                  merged.groupby(["split", "category"]).size().items()},
        "test_groups": sorted(test_groups),
    }
    save_json(summary, RESULTS_DIR / "dataset_summary.json")
    print(f"Wrote {DATASET_CSV} ({len(df)} rows) and {GROUPS_CSV}")
    print("Test groups:", sorted(test_groups))
    print(merged.groupby(["split", "label", "category"]).size())


if __name__ == "__main__":
    main()
