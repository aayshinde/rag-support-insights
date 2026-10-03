"""Harder, real-world stress tests for the confidence gate. None of this text came from the knowledge base.

  out-of-scope   CLINC150 `oos` queries: real user messages the support KB should never answer
  unrelated      CLINC150 in-scope queries about weather, travel, music, ... (a different domain)
  adjacent       Banking77 queries: real, noisy customer-support language from a neighbouring domain
  typos          the held-out tickets again, with character-level noise (swaps, drops, doubles)

Questions answered: does the gate escalate what it should never answer (AUROC against the in-domain
held-out tickets), and does the answered-set error survive noisy input?

Usage:  python -m src.stress_test
"""
from __future__ import annotations

import json
import random

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import config
from src.reliability import ConfidenceModel, add_typos, batch_features, embed_queries, load_eval_data, search_vectors

N = 300


def load_sets() -> dict[str, list[str]]:
    from datasets import load_dataset

    rng = random.Random(config.SEED)
    clinc = load_dataset("clinc/clinc_oos", "plus", split="test")
    names = clinc.features["intent"].names
    oos_id = names.index("oos")
    oos = [r["text"] for r in clinc if r["intent"] == oos_id]
    ins = [r["text"] for r in clinc if r["intent"] != oos_id]
    bank = [r["text"] for r in load_dataset("mteb/banking77", split="test")]
    return {
        "out-of-scope (CLINC oos)": rng.sample(oos, min(N, len(oos))),
        "unrelated domains (CLINC in-scope)": rng.sample(ins, N),
        "adjacent domain (Banking77)": rng.sample(bank, N),
    }


def gate_scores(store, gate, vecs):
    s, c = search_vectors(store, vecs, config.TOP_K)
    pred, X = batch_features(s, c)
    conf = gate.predict_proba(X)
    ood = (X[:, 0] < gate.ood_top1) if gate.ood_top1 is not None else np.zeros(len(conf), bool)
    return pred, X, conf, (conf < gate.tau) | ood


def cached_embed(name: str, texts: list[str]) -> np.ndarray:
    p = config.RESULTS_DIR / f"stress_emb_{name}_{len(texts)}_{config.EMBED_MODEL.replace(':', '_')}.npy"
    if p.exists():
        return np.load(p)
    v = embed_queries(texts)
    np.save(p, v)
    return v


def main() -> None:
    gate = ConfidenceModel.load()
    assert gate is not None, "run `python -m src.reliability` first"
    ev = load_eval_data(4000, config.TOP_K)
    store, test = ev["store"], ev["test"]
    pred_in, X_in, conf_in, esc_in = gate_scores(store, gate, ev["hold_vecs"])
    ok_in = np.array([p == g for p, g in zip(pred_in, test["category"])])

    out: dict = {"tau": gate.tau, "ood_top1": gate.ood_top1, "model": "gb",
           "in_domain": {"n": len(conf_in), "escalated": round(float(esc_in.mean()), 4), "escalated_corr_only": round(float((conf_in < gate.tau).mean()), 4)}, "sets": {}}
    feature_names = {"gate (gradient boosting)": None, "raw top-1 similarity": 0, "vote share": 3}
    for name, texts in load_sets().items():
        vecs = cached_embed(name.split()[0].lower() + str(len(name)), texts)
        _, X, conf, esc = gate_scores(store, gate, vecs)
        row = {"n": len(texts), "escalated": round(float(esc.mean()), 4), "escalated_corr_only": round(float((conf < gate.tau).mean()), 4), "mean_confidence": round(float(conf.mean()), 4), "auroc_vs_in_domain": {}}
        y = np.r_[np.ones(len(conf_in)), np.zeros(len(conf))]  # 1 = in-domain should be answered
        for label, col in feature_names.items():
            a = np.r_[conf_in, conf] if col is None else np.r_[X_in[:, col], X[:, col]]
            row["auroc_vs_in_domain"][label] = round(float(roc_auc_score(y, a)), 4)
        row["examples_answered_anyway"] = [t for t, e in zip(texts, esc) if not e][:3]
        out["sets"][name] = row

    rng = random.Random(config.SEED)
    noisy = [add_typos(q, rng) for q in test["query"]]
    pred_n, X_n, conf_n, esc_n = gate_scores(store, gate, cached_embed("typos", noisy))
    ok_n = np.array([p == g for p, g in zip(pred_n, test["category"])])
    ans_c, ans_n = ~esc_in, ~esc_n
    out["typos"] = {
        "rate": 0.08, "n": len(noisy), "example": next(([a[:80], b[:80]] for a, b in zip(test["query"], noisy) if a != b and len(a) < 80), ["", ""]),
        "clean": {"accuracy": round(float(ok_in.mean()), 4), "answered": round(float(ans_c.mean()), 4), "error_on_answered": round(float(1 - ok_in[ans_c].mean()), 4)},
        "noisy": {"accuracy": round(float(ok_n.mean()), 4), "answered": round(float(ans_n.mean()), 4), "error_on_answered": round(float(1 - ok_n[ans_n].mean()), 4)},
        "confidence_drop_mean": round(float((conf_in - conf_n).mean()), 4),
    }
    (config.RESULTS_DIR / "stress_test.json").write_text(json.dumps(out, indent=2))
    report(out)


def report(o: dict) -> None:
    import os

    def load_json(name):
        path = config.RESULTS_DIR / name
        return json.load(open(path)) if path.exists() else None

    before, aug = load_json("stress_test_before.json"), load_json("stress_test_aug.json")
    L = ["# Stress tests on real, out-of-distribution text", "",
         f"Two-stage gate: escalate if top-1 similarity < {o['ood_top1']:.3f} (relevance guard) or calibrated confidence < {o['tau']:.3f} (correctness gate). "
         f"In-domain held-out tickets are escalated {o['in_domain']['escalated']:.1%} of the time ({o['in_domain']['escalated_corr_only']:.1%} by the correctness gate alone).", "",
         "## Does the gate escalate what the knowledge base should never answer?", "",
         "| Query set | n | escalated (two-stage) | escalated (correctness gate only) | AUROC: gate | AUROC: raw top-1 | AUROC: vote share |", "|---|---|---|---|---|---|---|"]
    for k, v in o["sets"].items():
        a = v["auroc_vs_in_domain"]
        L.append(f"| {k} | {v['n']} | **{v['escalated']:.1%}** | {v['escalated_corr_only']:.1%} | {a['gate (gradient boosting)']:.3f} | {a['raw top-1 similarity']:.3f} | {a['vote share']:.3f} |")
    t = o["typos"]
    L += ["", "AUROC compares each out-of-domain set against the in-domain held-out tickets (1.0 = perfectly separated, 0.5 = chance).",
          "The correctness gate alone barely notices out-of-domain text, while raw similarity separates it almost perfectly. The two signals do different jobs, hence two stages.", "",
          f"## Robustness to typos (~{t['rate']:.0%} of letters dropped, doubled or substituted)", "",
          "| | accuracy (answer all) | answered | error on answered |", "|---|---|---|---|",
          f"| clean input | {t['clean']['accuracy']:.1%} | {t['clean']['answered']:.1%} | {t['clean']['error_on_answered']:.1%} |"]
    if before:
        b = before["typos"]["noisy"]
        L.append(f"| noisy, correctness gate only (first version) | {b['accuracy']:.1%} | {b['answered']:.1%} | {b['error_on_answered']:.1%} |")
    L += [f"| noisy, two-stage gate (deployed) | {t['noisy']['accuracy']:.1%} | {t['noisy']['answered']:.1%} | {t['noisy']['error_on_answered']:.1%} |"]
    if aug:
        a = aug["typos"]["noisy"]
        L.append(f"| noisy, two-stage gate trained with typo augmentation (tried, not kept) | {a['accuracy']:.1%} | {a['answered']:.1%} | {a['error_on_answered']:.1%} |")
    L += ["", "Typo augmentation did not reduce the error on answered queries, so the simpler gate trained on clean data is deployed. Robustness to noisy input remains an open weakness: 8% character noise costs about 6 points of accuracy and the gate does not fully compensate.", "",
          f"Example: `{t['example'][0]}` becomes `{t['example'][1]}`.", ""]
    (config.RESULTS_DIR / "stress_test.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
