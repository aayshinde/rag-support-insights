"""Latency and LLM-call savings benchmark. Run on an otherwise idle machine.

Measures, per query: embedding, FAISS search, gate, and end-to-end time, split by whether the gate
answered or escalated. Escalated queries skip the LLM, so the gate saves both latency and compute.
The traffic mix is in-domain held-out tickets plus out-of-domain messages (CLINC), because a
deployed support endpoint also receives text the knowledge base cannot answer.

Usage:  python -m src.benchmark --n-in 30 --n-out 15
"""
from __future__ import annotations

import argparse
import json
import platform
import time

import numpy as np
import pandas as pd

from src import config
from src.llm import get_embeddings
from src.rag import SupportRAG
from src.reliability import ConfidenceModel
from src.stress_test import load_sets


def pct(a, q):
    return round(float(np.percentile(a, q)), 3) if len(a) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-in", type=int, default=30)
    ap.add_argument("--n-out", type=int, default=15)
    args = ap.parse_args()

    gate = ConfidenceModel.load()
    rag = SupportRAG(gate=gate)
    emb = get_embeddings()
    test = pd.read_csv(config.TEST_CSV).sample(args.n_in, random_state=config.SEED)
    ood = load_sets()["out-of-scope (CLINC oos)"][: args.n_out]
    traffic = [(q, "in-domain") for q in test["query"]] + [(q, "out-of-domain") for q in ood]
    rag.run(traffic[0][0])  # warm up model load so it is not counted

    rows = []
    for q, kind in traffic:
        t0 = time.perf_counter(); v = emb.embed_query(q); t_emb = time.perf_counter() - t0
        t0 = time.perf_counter(); scored = rag.store.similarity_search_with_score_by_vector(v, k=config.TOP_K); t_search = time.perf_counter() - t0
        t0 = time.perf_counter()
        gate.decide([s for _, s in scored], [d.metadata["category"] for d, _ in scored]); t_gate = time.perf_counter() - t0
        t0 = time.perf_counter(); res = rag.run(q, "grounded"); t_total = time.perf_counter() - t0
        rows.append({"kind": kind, "abstained": res.abstained, "reason": res.abstain_reason, "embed_s": t_emb, "search_s": t_search, "gate_s": t_gate, "total_s": t_total})
    df = pd.DataFrame(rows)
    ans, esc = df[~df["abstained"]], df[df["abstained"]]
    out = {
        "machine": f"{platform.machine()} {platform.system()} (8 GB laptop, CPU/Metal, Ollama)", "n": len(df),
        "traffic": {"in_domain": args.n_in, "out_of_domain": args.n_out},
        "stage_median_ms": {"embed": round(df["embed_s"].median() * 1000, 1), "faiss_search": round(df["search_s"].median() * 1000, 2), "gate": round(df["gate_s"].median() * 1000, 2)},
        "answered": {"n": len(ans), "p50_s": pct(ans["total_s"], 50), "p95_s": pct(ans["total_s"], 95)},
        "escalated": {"n": len(esc), "p50_s": pct(esc["total_s"], 50), "p95_s": pct(esc["total_s"], 95)},
        "llm_calls_avoided": round(float(df["abstained"].mean()), 4),
        "mean_latency_s_with_gate": round(float(df["total_s"].mean()), 3),
        "mean_latency_s_if_every_query_called_llm": round(float(ans["total_s"].mean()), 3) if len(ans) else None,
        "escalated_by_kind": {k: round(float(g["abstained"].mean()), 3) for k, g in df.groupby("kind")},
    }
    (config.RESULTS_DIR / "benchmark.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
