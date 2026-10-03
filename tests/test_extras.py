"""Offline tests for the two-stage gate, conformal baseline, answer-quality helpers and failure analysis."""
from __future__ import annotations

import os

os.environ["EMBED_MODEL"] = "local"

import random

import numpy as np

from src.answer_eval import specifics, unsupported
from src.baselines import class_probs, conformal_sets
from src.failure_analysis import merge_map
from src.reliability import ConfidenceModel, add_typos, make_gate_model


def _gate(tau: float, ood: float | None):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 6))
    return ConfidenceModel(make_gate_model("gb").fit(X, X[:, 3] > 0), tau=tau, target_risk=0.1, ood_top1=ood)


def test_two_stage_gate_reports_the_reason():
    g = _gate(tau=-0.1, ood=0.5)  # correctness gate always passes
    assert g.decide([0.3, 0.2, 0.2], ["A", "A", "A"])[2] == "out_of_domain"  # nothing similar in the KB
    assert g.decide([0.9, 0.8, 0.8], ["A", "A", "A"])[2] is None
    strict = _gate(tau=1.1, ood=None)  # correctness gate never passes, relevance guard off
    assert strict.decide([0.9, 0.8, 0.8], ["A", "A", "A"])[2] == "low_confidence"


def test_old_gate_files_without_ood_threshold_still_load(tmp_path):
    import joblib

    g = _gate(0.5, 0.7)
    joblib.dump({"pipeline": g.pipeline, "tau": 0.5, "target_risk": 0.1}, tmp_path / "old.joblib")
    assert ConfidenceModel.load(tmp_path / "old.joblib").ood_top1 is None


def test_class_probs_sum_to_one_and_conformal_sets_grow_with_coverage():
    p = class_probs([0.9, 0.8, 0.3], ["A", "A", "B"])
    assert abs(sum(p.values()) - 1) < 1e-9 and p["A"] > p["B"]
    rng = np.random.default_rng(1)
    scores = [np.sort(rng.random(5))[::-1] for _ in range(300)]
    cats = [list(rng.choice(["A", "B", "C"], 5)) for _ in range(300)]
    gold = list(rng.choice(["A", "B", "C"], 300))
    tight = conformal_sets(scores, cats, gold, scores[:50], cats[:50], alpha=0.5)
    loose = conformal_sets(scores, cats, gold, scores[:50], cats[:50], alpha=0.05)
    assert sum(map(len, loose)) >= sum(map(len, tight))  # asking for more coverage never shrinks the sets


def test_unsupported_specifics_flags_invented_numbers_only():
    ans = "Refunds arrive within 48 hours. Call 555-0100 or see https://example.com/help."
    assert specifics(ans)
    total, bad = unsupported(ans, "customer wrote order 55123. refunds arrive within 48 hours")
    assert bad >= 2 and bad < total + 1  # the URL and the phone number are not in the sources
    assert unsupported("Refunds arrive within 48 hours.", "refunds arrive within 48 hours")[1] == 0
    assert unsupported("Please check your account.", "anything")[0] == 0


def test_merge_map_groups_transitively():
    m = merge_map([("IT", "TECH"), ("TECH", "PRODUCT"), ("A", "B")])
    assert m["IT"] == m["TECH"] == m["PRODUCT"] == "IT" and m["B"] == "A"


def test_add_typos_changes_text_but_keeps_non_letters():
    out = add_typos("refund my order #12345 now", random.Random(0), rate=0.5)
    assert out != "refund my order #12345 now" and "#12345" in out
