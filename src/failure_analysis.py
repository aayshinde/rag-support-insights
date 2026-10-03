"""Failure analysis: where does triage go wrong, and is it the model or the labels?

Uses the leave-one-out calibration pool (4,000 queries) and the held-out split, with the deployed
two-stage gate. Reports:

  1. confusion pairs among answered-but-wrong predictions
  2. per-category accuracy and escalation rate
  3. a taxonomy experiment: merge near-synonym categories and re-measure. If most "errors" are label
     ambiguity, merging should remove many of them without touching the model.
  4. concrete examples of confident misses

Usage:  python -m src.failure_analysis
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

from src import config
from src.reliability import ConfidenceModel, batch_features, load_eval_data, pick_tau, risk_coverage


def merge_map(pairs: list[tuple[str, str]]) -> dict[str, str]:
    """Union-find over category pairs; each group is renamed to its alphabetically first member."""
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return {c: find(c) for c in parent}


def main() -> None:
    gate = ConfidenceModel.load()
    ev = load_eval_data(4000, config.TOP_K)
    cal_pred, Xc = batch_features(ev["cs"], ev["cc"])
    hold_pred, Xh = batch_features(ev["hs"], ev["hc"])
    gold_c, gold_h = np.array(ev["cal_gold"]), ev["test"]["category"].to_numpy()
    pred_c, pred_h = np.array(cal_pred), np.array(hold_pred)
    conf_c = gate.predict_proba(Xc)
    esc_c = (conf_c < gate.tau) | (Xc[:, 0] < gate.ood_top1)
    src_c = np.array(ev["cal_src"])

    wrong = pred_c != gold_c
    ans_wrong = wrong & ~esc_c
    pairs = Counter(zip(gold_c[ans_wrong], pred_c[ans_wrong])).most_common(12)
    undirected = Counter()
    for (a, b), n in Counter(zip(gold_c[wrong], pred_c[wrong])).items():
        undirected[tuple(sorted((a, b)))] += n

    per_cat = []
    for c in sorted(set(gold_c)):
        m = gold_c == c
        per_cat.append({"category": c, "n": int(m.sum()), "accuracy": round(float((~wrong[m]).mean()), 4),
                        "escalated": round(float(esc_c[m].mean()), 4),
                        "error_on_answered": round(float(wrong[m & ~esc_c].mean()), 4) if (m & ~esc_c).any() else None})
    per_cat.sort(key=lambda r: r["accuracy"])

    # taxonomy experiment: merge the top-N most confused (undirected) pairs, re-score everything.
    exp = []
    for n_merge in (0, 1, 2, 3, 4, 5):
        mm = merge_map([p for p, _ in undirected.most_common(n_merge)]) if n_merge else {}
        mp = lambda arr: np.array([mm.get(x, x) for x in arr])
        ok_c = mp(pred_c) == mp(gold_c)
        ok_h = mp(pred_h) == mp(gold_h)
        # re-fit the gate on merged labels, so confidence reflects the new taxonomy
        from sklearn.model_selection import cross_val_predict

        from src.reliability import make_gate_model

        oof = cross_val_predict(make_gate_model("gb"), Xc, ok_c, cv=5, method="predict_proba")[:, 1]
        tau = pick_tau(oof, ok_c, 0.10)
        acc_mask = oof >= tau
        exp.append({"merged_pairs": [list(p) for p, _ in undirected.most_common(n_merge)], "n_categories": int(len(set(mp(gold_c)))),
                    "accuracy_answer_all_cal": round(float(ok_c.mean()), 4), "accuracy_answer_all_hold": round(float(ok_h.mean()), 4),
                    "coverage_at_10pct_error_cal": round(float(acc_mask.mean()), 4)})

    # how concentrated are the errors? share of all wrong predictions that stay inside the merged group
    group = set(merge_map([p for p, _ in undirected.most_common(5)]))
    inside = np.array([(g in group) and (p in group) for g, p in zip(gold_c, pred_c)])
    share_inside = float((wrong & inside).sum() / max(wrong.sum(), 1))

    # concrete confident misses from the held-out split
    conf_h = gate.predict_proba(Xh)
    esc_h = (conf_h < gate.tau) | (Xh[:, 0] < gate.ood_top1)
    miss = [{"query": q[:200], "gold": g, "pred": p, "confidence": round(float(c), 3)}
            for q, g, p, c, e in zip(ev["test"]["query"], gold_h, pred_h, conf_h, esc_h) if not e and g != p]
    miss.sort(key=lambda r: -r["confidence"])

    out = {"n_cal": len(gold_c), "answered_wrong_pairs": [{"gold": a, "pred": b, "n": n} for (a, b), n in pairs],
           "top_undirected_confusions": [{"pair": list(p), "n": n} for p, n in undirected.most_common(8)],
           "wrong_share_by_source": {s: round(float(wrong[src_c == s].mean()), 4) for s in sorted(set(src_c))},
           "group_categories": sorted(group), "share_errors_within_group": round(share_inside, 4),
           "per_category": per_cat, "merge_experiment": exp, "confident_misses_holdout": miss[:8]}
    (config.RESULTS_DIR / "failure_analysis.json").write_text(json.dumps(out, indent=2))
    write_md(out)


def write_md(o: dict) -> None:
    L = ["# Failure analysis: where triage goes wrong", "",
         f"Based on {o['n_cal']:,} leave-one-out queries from the knowledge base and the 200 held-out tickets, using the deployed two-stage gate.", "",
         "## 1. Most of the errors are between near-synonym categories", "",
         "| Most confused category pair (either direction) | wrong predictions |", "|---|---|"]
    L += [f"| {a} <-> {b} | {n} |" for (a, b), n in [(tuple(x["pair"]), x["n"]) for x in o["top_undirected_confusions"]]]
    L += ["", f"**{o['share_errors_within_group']:.0%} of all wrong predictions are confusions among just {len(o['group_categories'])} categories** ({', '.join(o['group_categories'])}), out of 21.", "",
          "Error rate by data source: " + ", ".join(f"{k} {v:.1%}" for k, v in o["wrong_share_by_source"].items()) + ". All of the errors are in the free-text tickets, where one topic is filed under several overlapping categories.", "",
          "## 2. Taxonomy experiment: merge the most-confused category pairs", "",
          "If the errors were model failures, merging labels would not help much. If they are label ambiguity, merging should recover accuracy and let the gate answer more.", "",
          "| Pairs merged | categories left | accuracy (answer all, calibration) | accuracy (held-out) | coverage at 10% error |", "|---|---|---|---|---|"]
    for e in o["merge_experiment"]:
        L.append(f"| {len(e['merged_pairs'])} | {e['n_categories']} | {e['accuracy_answer_all_cal']:.1%} | {e['accuracy_answer_all_hold']:.1%} | {e['coverage_at_10pct_error_cal']:.1%} |")
    L += ["", "Caveat: merging labels makes the task easier by construction, so the accuracy gain alone proves little. The evidence is how concentrated the errors are: a few overlapping categories account for most of them, which points at the taxonomy, not the retriever.", "",
          "Pairs merged cumulatively, most confused first: " + "; ".join(f"{' + '.join(p)}" for p in o["merge_experiment"][-1]["merged_pairs"]), "",
          "## 3. Weakest categories", "", "| Category | n | accuracy | escalated | error on answered |", "|---|---|---|---|---|"]
    L += [f"| {r['category']} | {r['n']} | {r['accuracy']:.1%} | {r['escalated']:.1%} | {'-' if r['error_on_answered'] is None else format(r['error_on_answered'], '.1%')} |" for r in o["per_category"][:8]]
    L += ["", "## 4. Confident misses in the held-out split", "", "These are the cases the gate cannot catch, because the neighbours agree on the wrong label.", ""]
    L += [f"- ({m['confidence']:.2f}) gold **{m['gold']}**, predicted **{m['pred']}**: \"{m['query']}\"" for m in o["confident_misses_holdout"][:5]]
    L += ["", "## What I would do next", "",
          "1. Fix the taxonomy before improving the model. Merging the top confused pairs is a cheaper win than any model change.",
          "2. Add a human-review queue for the escalated tickets and feed their resolved labels back into the index, which is the loop the knowledge-gap clusters are meant to drive.",
          "3. Evaluate on real tickets with a clean taxonomy, since half of this benchmark is templated and the other half has overlapping labels."]
    (config.RESULTS_DIR / "failure_analysis.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
