"""Offline end-to-end test: synthetic data -> local embeddings -> FAISS -> RAG -> judge.
No API key or network needed.  Run:  pytest -q
"""
from __future__ import annotations

import os

os.environ["EMBED_MODEL"] = "local"

import pandas as pd
import pytest
from langchain_core.runnables import RunnableLambda

from src import config, data
from src.evaluate import FaithfulnessJudgement, Judge, RelevanceJudgement, evaluate_one, summarise
from src.ingest import build_index
from src.rag import SupportInsight, SupportRAG, knn_vote


class FakeLLM:
    """Stands in for ChatOpenAI: returns a schema instance derived from the prompt text."""

    def with_structured_output(self, schema):
        def respond(prompt_value):
            text = prompt_value.to_string()
            if schema is SupportInsight:
                cat = "REFUND" if "refund" in text.lower() else "ORDER"
                return SupportInsight(
                    category=cat, intent="get_refund", next_best_action="Submit a refund request under Returns.",
                    steps=["Open Returns", "Submit the request"], cited_sources=[1], confidence=0.8,
                )
            if schema is FaithfulnessJudgement:
                return FaithfulnessJudgement(claims=[
                    {"claim": "Submit a refund request under Returns", "supported": "Returns" in text},
                    {"claim": "Refund arrives in 2 hours", "supported": False},
                ])
            if schema is RelevanceJudgement:
                return RelevanceJudgement(score=4, reason="addresses refund")
            raise TypeError(schema)

        return RunnableLambda(respond)


@pytest.fixture(scope="module")
def corpus():
    df = data.clean(data.make_synthetic(600))
    return data.split(df, n_test=30)


@pytest.fixture(scope="module")
def rag(corpus):
    kb, _ = corpus
    store = build_index(kb)
    return SupportRAG(store=store, llm=FakeLLM(), categories=sorted(kb["category"].unique()), gate=None)


def test_split_has_no_leakage(corpus):
    kb, test = corpus
    assert len(test) == 30
    assert not set(kb["query"]) & set(test["query"])


def test_retrieval_finds_same_intent(rag):
    scored = rag.retrieve("I need a refund for my order #12345", "grounded")
    assert scored and scored[0][0].metadata["intent"] == "get_refund"
    assert all(-1.0 <= s <= 1.0001 for _, s in scored)  # cosine scale
    assert knn_vote(scored) == "REFUND"


def test_grounded_threshold_drops_weak_matches(rag):
    naive = rag.retrieve("zzzz qqqq", "naive")
    grounded = rag.retrieve("zzzz qqqq", "grounded")
    assert len(grounded) <= len(naive) and len(grounded) >= 1


def test_run_returns_structured_result(rag):
    res = rag.run("please refund order #55555", "grounded")
    assert res.insight["category"] == "REFUND"
    assert res.contexts and res.retrieved and "Next best action" in res.answer


def test_evaluate_one_and_summary(rag, corpus):
    _, test = corpus
    judge = Judge(llm=FakeLLM())
    recs = [evaluate_one(rag, judge, r, "grounded") for _, r in test.head(5).iterrows()]
    s = summarise(pd.DataFrame(recs))
    assert s["n"] == 5
    assert 0 <= s["mean_faithfulness"] <= 1
    assert s["relevance_rate"] == 1.0
    assert recs[0]["faithfulness"] == 0.5 and recs[0]["hallucination"] == (0.5 < config.HALLUCINATION_THRESHOLD)
