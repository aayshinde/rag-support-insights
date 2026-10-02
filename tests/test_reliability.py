"""Offline tests for the reliability layer (confidence gate, selective-prediction metrics, gaps)."""
from __future__ import annotations

import os

os.environ["EMBED_MODEL"] = "local"

import numpy as np
import pytest
from langchain_core.runnables import RunnableLambda

from src import data
from src.ingest import build_index
from src.rag import SupportInsight, SupportRAG
from src.reliability import (
    ConfidenceModel,
    aurc,
    ece,
    find_gaps,
    make_pipeline_model,
    pick_tau,
    risk_coverage,
    vote_features,
)


def test_vote_features_unanimous_vs_split():
    w1, f1 = vote_features([0.9, 0.8, 0.7], ["A", "A", "A"])
    w2, f2 = vote_features([0.9, 0.8, 0.7], ["A", "B", "C"])
    assert w1 == "A" and f1[3] == pytest.approx(1.0) and f1[4] == pytest.approx(1.0) and f1[5] == pytest.approx(0, abs=1e-6)
    assert w2 == "A" and f2[3] < 0.5 and f2[5] > f1[5]  # split vote -> lower share, higher entropy


def test_vote_features_handles_non_positive_scores():
    w, f = vote_features([-0.1, -0.2], ["A", "B"])
    assert w in {"A", "B"} and np.isfinite(f).all()


def test_risk_coverage_is_sorted_and_perfect_ranking_has_low_aurc():
    conf = np.array([0.9, 0.8, 0.3, 0.2])
    correct = np.array([True, True, False, False])
    rc = risk_coverage(conf, correct)
    assert rc["coverage"].iloc[-1] == 1.0 and rc["risk"].iloc[0] == 0.0 and rc["risk"].iloc[1] == 0.0
    assert aurc(conf, correct) < aurc(-conf, correct)


def test_pick_tau_respects_target_risk():
    rng = np.random.default_rng(0)
    conf = rng.random(2000)
    correct = rng.random(2000) < conf  # well calibrated
    tau = pick_tau(conf, correct, target_risk=0.10)
    accepted = conf >= tau
    assert accepted.any() and 1 - correct[accepted].mean() <= 0.10 + 1e-9
    assert pick_tau(conf, correct, target_risk=0.0) >= 0.0


def test_ece_is_small_when_calibrated_and_large_when_not():
    rng = np.random.default_rng(1)
    conf = rng.random(5000)
    assert ece(conf, rng.random(5000) < conf) < 0.05
    assert ece(np.full(5000, 0.95), rng.random(5000) < 0.2) > 0.5


def test_confidence_model_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 6))
    y = X[:, 3] + rng.normal(scale=0.3, size=300) > 0
    cm = ConfidenceModel(make_pipeline_model().fit(X, y), tau=0.5, target_risk=0.1)
    path = tmp_path / "c.joblib"
    cm.save(path)
    loaded = ConfidenceModel.load(path)
    assert loaded.tau == 0.5 and np.allclose(loaded.predict_proba(X), cm.predict_proba(X))
    assert ConfidenceModel.load(tmp_path / "missing.joblib") is None


def test_find_gaps_clusters_low_confidence_queries():
    rng = np.random.default_rng(3)
    a, b = rng.normal(0, 0.1, (40, 8)) + 1, rng.normal(0, 0.1, (40, 8)) - 1
    vecs = np.vstack([a, b])
    texts = ["security breach on medical data"] * 40 + ["marketing campaign brand growth"] * 40
    gaps = find_gaps(texts, vecs, np.zeros(80), ["X"] * 80, tau=0.5, max_clusters=2)
    assert len(gaps) == 2 and sum(g["size"] for g in gaps) == 80
    topics = {" ".join(g["keywords"]).split()[0] for g in gaps}
    assert any(t in {"security", "breach", "medical", "data"} for t in topics)
    assert any(t in {"marketing", "campaign", "brand", "growth"} for t in topics)


class CountingLLM:
    def __init__(self):
        self.calls = 0

    def with_structured_output(self, schema):
        def respond(_):
            self.calls += 1
            return SupportInsight(category="REFUND", intent="get_refund", next_best_action="Refund it.", steps=["a", "b"], confidence=0.9)

        return RunnableLambda(respond)


@pytest.fixture(scope="module")
def store_and_cats():
    kb, _ = data.split(data.clean(data.make_synthetic(400)), n_test=20)
    return build_index(kb), sorted(kb["category"].unique())


def _gate(tau: float) -> ConfidenceModel:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6))
    return ConfidenceModel(make_pipeline_model().fit(X, X[:, 0] > 0), tau=tau, target_risk=0.1)


def test_gate_abstains_without_calling_llm(store_and_cats):
    store, cats = store_and_cats
    llm = CountingLLM()
    rag = SupportRAG(store=store, llm=llm, categories=cats, gate=_gate(tau=1.1))  # nothing can reach 1.1
    res = rag.run("please refund order #55555", "grounded")
    assert res.abstained and res.insight["escalate"] and llm.calls == 0
    assert 0 <= res.calibrated_confidence <= 1 and "Escalate" in res.answer


def test_gate_passes_confident_queries_and_never_gates_naive(store_and_cats):
    store, cats = store_and_cats
    llm = CountingLLM()
    rag = SupportRAG(store=store, llm=llm, categories=cats, gate=_gate(tau=-0.1))  # everything passes
    assert not rag.run("refund please", "grounded").abstained and llm.calls == 1
    strict = SupportRAG(store=store, llm=llm, categories=cats, gate=_gate(tau=1.1))
    assert not strict.run("refund please", "naive").abstained  # baseline stays ungated for comparison
