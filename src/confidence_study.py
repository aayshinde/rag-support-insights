"""Does the LLM's self-reported confidence predict its own errors better than retrieval evidence?

Runs the grounded pipeline (gate disabled) on held-out queries and, for each, records
  - the LLM's verbalised confidence,
  - the calibrated retrieval-based confidence from src.reliability,
  - whether the LLM's predicted category matches the gold label.
Then compares how well each score separates correct from incorrect answers (AUROC, AURC).
Results append to results/confidence_study.jsonl, so an interrupted run resumes.

Usage:  python -m src.confidence_study --n 100
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src import config
from src.rag import SupportRAG
from src.reliability import ConfidenceModel, aurc

PATH = config.RESULTS_DIR / "confidence_study.jsonl"


def norm(s) -> str:
    return str(s).strip().lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    gate = ConfidenceModel.load()
    assert gate is not None, "run `python -m src.reliability` first"
    rag = SupportRAG(gate=None)
    test = pd.read_csv(config.TEST_CSV).head(args.n)
    done = {json.loads(l)["ticket_id"] for l in PATH.read_text().splitlines()} if PATH.exists() else set()

    with PATH.open("a") as f:
        for _, row in tqdm(test[~test["ticket_id"].isin(done)].iterrows(), total=len(test) - len(done)):
            try:
                res = rag.run(row["query"], "grounded")
            except Exception as e:  # small local models sometimes emit invalid JSON; skip and move on
                tqdm.write(f"  ! {row['ticket_id']}: {type(e).__name__}")
                continue
            full = rag.store.similarity_search_with_score(row["query"], k=rag.top_k)
            _, cal = gate.confidence([s for _, s in full], [d.metadata["category"] for d, _ in full])
            rec = {
                "ticket_id": row["ticket_id"],
                "source": row["source"],
                "llm_correct": norm(res.insight["category"]) == norm(row["category"]),
                "knn_correct": norm(res.knn_category) == norm(row["category"]),
                "llm_confidence": res.insight["confidence"],
                "llm_escalate": res.insight["escalate"],
                "calibrated_confidence": cal,
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()

    df = pd.read_json(PATH, lines=True)
    out = {"n": int(len(df)), "llm_category_accuracy": round(float(df["llm_correct"].mean()), 4),
           "knn_category_accuracy": round(float(df["knn_correct"].mean()), 4), "signals": {}}
    y = df["llm_correct"].to_numpy()
    for name, col in [("llm_verbalised_confidence", "llm_confidence"), ("retrieval_calibrated_confidence", "calibrated_confidence")]:
        c = df[col].to_numpy()
        out["signals"][name] = {
            "auroc": round(float(roc_auc_score(y, c)), 4) if 0 < y.mean() < 1 else None,
            "aurc": round(aurc(c, y), 4),
            "distinct_values": int(len(np.unique(c))),
        }
    (config.RESULTS_DIR / "confidence_study.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
