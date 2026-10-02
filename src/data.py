"""Download, normalise and split the customer-support corpus.

Normalised schema (one row per support record):
    ticket_id, query, category, intent, resolution, source

Sources
-------
bitext   bitext/Bitext-customer-support-llm-chatbot-training-dataset (~26.9K rows,
         11 categories / 27 intents, every row has an agent response)
tickets  Tobi-Bueck/customer-support-tickets (English rows only; queue -> category,
         type -> intent, answer -> resolution)
both     bitext + tickets concatenated  (this is how you get past 28K records)
csv      your own CSV with the normalised columns (pass --csv path)

Usage
-----
    python -m src.data --source both
    python -m src.data --source csv --csv my_tickets.csv
    python -m src.data --source synthetic --n 3000      # offline smoke test only
"""
from __future__ import annotations

import argparse
import random

import pandas as pd

from src import config

COLUMNS = ["ticket_id", "query", "category", "intent", "resolution", "source"]


def load_bitext() -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
    df = ds.to_pandas()
    out = pd.DataFrame(
        {
            "query": df["instruction"],
            "category": df["category"].str.upper(),
            "intent": df["intent"],
            "resolution": df["response"],
            "source": "bitext",
        }
    )
    return out


def load_tickets() -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
    df = ds.to_pandas()
    if "language" in df.columns:
        df = df[df["language"] == "en"]
    subject = df["subject"].fillna("") if "subject" in df.columns else ""
    out = pd.DataFrame(
        {
            "query": (subject + "\n" + df["body"].fillna("")).str.strip(),
            "category": df["queue"].fillna("General Inquiry").str.upper(),
            "intent": df["type"].fillna("Request"),
            "resolution": df["answer"].fillna(""),
            "source": "tickets",
        }
    )
    return out


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"query", "category", "resolution"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    if "intent" not in df.columns:
        df["intent"] = df["category"]
    df["source"] = df.get("source", "csv")
    return df


def make_synthetic(n: int, seed: int = config.SEED) -> pd.DataFrame:
    """Small templated corpus so the pipeline can be exercised without network access.
    NOT a substitute for real data — never report metrics computed on it."""
    rng = random.Random(seed)
    catalog = {
        ("ORDER", "cancel_order"): (
            ["I want to cancel my order {o}", "how do I cancel order {o}?", "please cancel purchase {o}"],
            "Open Order History, select order {o} and click Cancel. Orders that have shipped must be returned instead.",
        ),
        ("ORDER", "track_order"): (
            ["where is my order {o}", "track package for order {o}", "my order {o} hasn't arrived"],
            "Use the tracking link in your confirmation email or check Order History for order {o}. Allow 48h for carrier updates.",
        ),
        ("REFUND", "get_refund"): (
            ["I need a refund for {o}", "how can I get my money back for order {o}", "refund request {o}"],
            "Submit a refund request under Returns for order {o}. Refunds post to the original payment method in 5-7 business days.",
        ),
        ("ACCOUNT", "recover_password"): (
            ["I forgot my password", "can't log in, reset password", "password reset link not working"],
            "Click 'Forgot password' on the sign-in page. If the link has expired, request a new one; links are valid for 30 minutes.",
        ),
        ("PAYMENT", "payment_issue"): (
            ["my card was declined", "payment failed on checkout", "charged twice for order {o}"],
            "Verify the billing address matches your card. Duplicate charges are pending holds and drop off within 3 business days.",
        ),
        ("SHIPPING", "change_shipping_address"): (
            ["change delivery address for {o}", "I moved, update shipping address", "wrong address on order {o}"],
            "Shipping address can be edited in Order History until the order ships. After that, contact the carrier to redirect.",
        ),
    }
    keys = list(catalog)
    rows = []
    for _ in range(n):
        cat, intent = rng.choice(keys)
        templates, resolution = catalog[(cat, intent)]
        o = f"#{rng.randint(10000, 99999)}"
        rows.append(
            {
                "query": rng.choice(templates).format(o=o),
                "category": cat,
                "intent": intent,
                "resolution": resolution.format(o=o),
                "source": "synthetic",
            }
        )
    return pd.DataFrame(rows)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("query", "resolution"):
        df[col] = df[col].astype(str).str.strip()
    df = df[(df["query"].str.len() > 3) & (df["resolution"].str.len() > 3)]
    df = df.drop_duplicates(subset=["query"]).reset_index(drop=True)
    df["ticket_id"] = [f"T{i:06d}" for i in range(len(df))]
    return df[COLUMNS]


def split(df: pd.DataFrame, n_test: int, seed: int = config.SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified hold-out: the eval queries are never embedded, so retrieval can't
    just find the exact answer (that would inflate every metric)."""
    from sklearn.model_selection import train_test_split

    strat = df["category"] if df["category"].value_counts().min() >= 2 else None
    kb, test = train_test_split(df, test_size=n_test, random_state=seed, stratify=strat)
    return kb.reset_index(drop=True), test.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["bitext", "tickets", "both", "csv", "synthetic"], default="both")
    ap.add_argument("--csv", help="path for --source csv")
    ap.add_argument("--n", type=int, default=3000, help="rows for --source synthetic")
    ap.add_argument("--n-test", type=int, default=config.N_EVAL_QUERIES)
    args = ap.parse_args()

    if args.source == "bitext":
        df = load_bitext()
    elif args.source == "tickets":
        df = load_tickets()
    elif args.source == "both":
        df = pd.concat([load_bitext(), load_tickets()], ignore_index=True)
    elif args.source == "csv":
        df = load_csv(args.csv)
    else:
        df = make_synthetic(args.n)

    df = clean(df)
    kb, test = split(df, args.n_test)
    df.to_csv(config.RAW_CSV, index=False)
    kb.to_csv(config.KB_CSV, index=False)
    test.to_csv(config.TEST_CSV, index=False)

    print(f"Total records : {len(df):,}")
    print(f"Indexed (KB)  : {len(kb):,}")
    print(f"Held-out eval : {len(test):,}")
    print("Categories    :", df["category"].nunique(), "|", ", ".join(sorted(df["category"].unique())[:12]))


if __name__ == "__main__":
    main()
