"""RAGAS-inspired evaluation of the RAG pipeline on held-out support queries.

Metrics (per query, then aggregated)
------------------------------------
faithfulness      RAGAS method: an LLM judge splits the answer into atomic claims and
                  checks each one against the retrieved context. score = supported / total.
hallucination     faithfulness < HALLUCINATION_THRESHOLD (default 0.7).
relevance         LLM judge rates 1-5 how well the answer addresses the customer's
                  message; "relevant" = score >= 4. relevance_rate = share of relevant answers.
category_acc      predicted category == gold category of the held-out record.
intent_acc        predicted intent == gold intent.
knn_category_acc  similarity-weighted vote over retrieved neighbours (non-LLM baseline).
retrieval_hit@k   at least one retrieved record shares the gold intent.

Both pipeline variants are run on the same queries, so the hallucination reduction
is a like-for-like comparison:  (rate_naive - rate_grounded) / rate_naive.

Usage:
    python -m src.evaluate                 # 200 queries, both variants
    python -m src.evaluate --n 20          # quick, cheap check
    python -m src.evaluate --variants grounded
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from tqdm import tqdm

from src import config
from src.llm import get_chat
from src.rag import SupportRAG

# ---------------------------------------------------------------- judges


class ClaimVerdict(BaseModel):
    claim: str
    supported: bool = Field(description="True only if the context explicitly supports the claim")


class FaithfulnessJudgement(BaseModel):
    claims: list[ClaimVerdict]


class RelevanceJudgement(BaseModel):
    score: int = Field(ge=1, le=5)
    reason: str


FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You verify whether an AI support answer is grounded in the provided context.\n"
            "Step 1: break the ANSWER into short atomic factual claims (ignore the category label "
            "line and generic advice like 'apologise to the customer').\n"
            "Step 2: for each claim, mark supported=true only if the CONTEXT states or directly "
            "implies it. Specific timeframes, amounts, policies, links or features not in the "
            "context are NOT supported.",
        ),
        ("human", "CONTEXT:\n{context}\n\nANSWER:\n{answer}"),
    ]
)

RELEVANCE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rate how well the support recommendation addresses the customer's message.\n"
            "5 = directly resolves the customer's actual issue with actionable steps\n"
            "4 = addresses the issue, minor gaps\n"
            "3 = partially relevant or generic\n"
            "2 = mostly off-target\n"
            "1 = wrong issue entirely\n"
            "Judge relevance only, not factual accuracy.",
        ),
        ("human", "CUSTOMER MESSAGE:\n{query}\n\nRECOMMENDATION:\n{answer}"),
    ]
)


class Judge:
    def __init__(self, llm=None):
        llm = llm if llm is not None else get_chat(config.JUDGE_MODEL)
        self.faith = FAITHFULNESS_PROMPT | llm.with_structured_output(FaithfulnessJudgement)
        self.rel = RELEVANCE_PROMPT | llm.with_structured_output(RelevanceJudgement)

    def faithfulness(self, answer: str, contexts: list[str]) -> tuple[float, list[dict]]:
        out: FaithfulnessJudgement = self.faith.invoke({"answer": answer, "context": "\n\n".join(contexts)})
        claims = [c.model_dump() for c in out.claims]
        if not claims:
            return 1.0, claims
        return sum(c["supported"] for c in claims) / len(claims), claims

    def relevance(self, query: str, answer: str) -> tuple[int, str]:
        out: RelevanceJudgement = self.rel.invoke({"query": query, "answer": answer})
        return out.score, out.reason


# ---------------------------------------------------------------- runner


def norm(s) -> str:
    return str(s).strip().lower().replace(" ", "_")


def evaluate_one(rag: SupportRAG, judge: Judge, row: pd.Series, variant: str) -> dict:
    t0 = time.time()
    res = rag.run(row["query"], variant=variant)
    latency = time.time() - t0
    faith, claims = judge.faithfulness(res.answer, res.contexts)
    rel, rel_reason = judge.relevance(row["query"], res.answer)
    gold_intent = norm(row["intent"])
    return {
        "ticket_id": row["ticket_id"],
        "variant": variant,
        "query": row["query"],
        "gold_category": row["category"],
        "gold_intent": row["intent"],
        "pred_category": res.insight["category"],
        "pred_intent": res.insight["intent"],
        "knn_category": res.knn_category,
        "answer": res.answer,
        "escalate": res.insight["escalate"],
        "confidence": res.insight["confidence"],
        "n_contexts": len(res.contexts),
        "top_score": res.retrieved[0]["score"] if res.retrieved else None,
        "retrieval_hit": any(norm(r["intent"]) == gold_intent for r in res.retrieved),
        "faithfulness": faith,
        "claims": claims,
        "hallucination": faith < config.HALLUCINATION_THRESHOLD,
        "relevance_score": rel,
        "relevant": rel >= 4,
        "relevance_reason": rel_reason,
        "category_correct": norm(res.insight["category"]) == norm(row["category"]),
        "intent_correct": norm(res.insight["intent"]) == gold_intent,
        "knn_correct": norm(res.knn_category) == norm(row["category"]),
        "latency_s": round(latency, 2),
    }


def run_variant(rag, judge, test: pd.DataFrame, variant: str, workers: int) -> pd.DataFrame:
    """Results are appended to a JSONL file as they finish, so an interrupted run resumes."""
    path = config.RESULTS_DIR / f"eval_{variant}.jsonl"
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            done[rec["ticket_id"]] = rec
    todo = test[~test["ticket_id"].isin(done)]

    with path.open("a") as f, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(evaluate_one, rag, judge, r, variant): r["ticket_id"] for _, r in todo.iterrows()}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"eval[{variant}]"):
            try:
                rec = fut.result()
            except Exception as e:  # keep going; failed rows are retried on the next run
                tqdm.write(f"  ! {futures[fut]}: {type(e).__name__}: {e}")
                continue
            done[rec["ticket_id"]] = rec
            f.write(json.dumps(rec) + "\n")
            f.flush()

    return pd.DataFrame([done[t] for t in test["ticket_id"] if t in done])


def summarise(df: pd.DataFrame) -> dict:
    return {
        "n": int(len(df)),
        "relevance_rate": round(df["relevant"].mean(), 4),
        "mean_relevance_1to5": round(df["relevance_score"].mean(), 3),
        "mean_faithfulness": round(df["faithfulness"].mean(), 4),
        "hallucination_rate": round(df["hallucination"].mean(), 4),
        "category_acc": round(df["category_correct"].mean(), 4),
        "intent_acc": round(df["intent_correct"].mean(), 4),
        "knn_category_acc": round(df["knn_correct"].mean(), 4),
        "retrieval_hit_at_k": round(df["retrieval_hit"].mean(), 4),
        "escalation_rate": round(df["escalate"].mean(), 4),
        "p50_latency_s": round(df["latency_s"].median(), 2),
    }


def write_report(summary: dict, path: Path) -> None:
    v = summary["variants"]
    rows = [
        ("Relevance rate (judge >= 4/5)", "relevance_rate", True),
        ("Mean relevance (1-5)", "mean_relevance_1to5", False),
        ("Mean faithfulness", "mean_faithfulness", True),
        ("Hallucination rate", "hallucination_rate", True),
        ("Category accuracy (LLM)", "category_acc", True),
        ("Category accuracy (kNN vote)", "knn_category_acc", True),
        ("Intent accuracy", "intent_acc", True),
        ("Retrieval hit@k (intent)", "retrieval_hit_at_k", True),
        ("Escalation rate", "escalation_rate", True),
        ("p50 latency (s)", "p50_latency_s", False),
    ]
    names = list(v)
    lines = [
        "# Evaluation report",
        "",
        f"- Held-out queries: **{summary['n_queries']}** (never indexed)",
        f"- Chat model: `{summary['chat_model']}`, judge: `{summary['judge_model']}`, embeddings: `{summary['embed_model']}`",
        f"- Hallucination = faithfulness < {summary['hallucination_threshold']}",
        "",
        "| Metric | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for label, key, pct in rows:
        cells = [f"{v[n][key]:.1%}" if pct else f"{v[n][key]}" for n in names]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    if "hallucination_reduction" in summary:
        lines += ["", f"**Relative hallucination reduction (naive → grounded): {summary['hallucination_reduction']:.1%}**"]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=config.N_EVAL_QUERIES)
    ap.add_argument("--variants", nargs="+", default=["naive", "grounded"], choices=["naive", "grounded"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fresh", action="store_true", help="ignore cached results and re-run")
    args = ap.parse_args()

    test = pd.read_csv(config.TEST_CSV).head(args.n)
    if args.fresh:
        for v in args.variants:
            (config.RESULTS_DIR / f"eval_{v}.jsonl").unlink(missing_ok=True)

    rag, judge = SupportRAG(), Judge()
    summary = {
        "n_queries": len(test),
        "chat_model": config.CHAT_MODEL,
        "judge_model": config.JUDGE_MODEL,
        "embed_model": config.EMBED_MODEL,
        "top_k": config.TOP_K,
        "min_similarity": config.MIN_SIMILARITY,
        "hallucination_threshold": config.HALLUCINATION_THRESHOLD,
        "variants": {},
    }
    for v in args.variants:
        df = run_variant(rag, judge, test, v, args.workers)
        summary["variants"][v] = summarise(df)

    if {"naive", "grounded"} <= summary["variants"].keys():
        a = summary["variants"]["naive"]["hallucination_rate"]
        b = summary["variants"]["grounded"]["hallucination_rate"]
        summary["hallucination_reduction"] = round((a - b) / a, 4) if a else 0.0

    (config.RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    write_report(summary, config.RESULTS_DIR / "report.md")
    print((config.RESULTS_DIR / "report.md").read_text())


if __name__ == "__main__":
    main()
