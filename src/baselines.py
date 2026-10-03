"""Baselines for the confidence gate: is the added machinery actually earning its place?

Compared on the same leave-one-out calibration pool and the same held-out split as src.reliability.

  raw top-1 similarity      threshold the best match's cosine score
  vote share                threshold the winning category's share of similarity mass
  entropy / agreement       single-feature thresholds
  split conformal (LAC)     class probabilities from the weighted vote; abstain unless the
                            conformal prediction set is a single category. Has a finite-sample
                            coverage guarantee, but gives one operating point per alpha.
  gradient boosting         HistGradientBoosting on the same six features
  logistic + isotonic       the deployed model, with and without isotonic recalibration
  logistic (deployed)       src.reliability's model

Usage:  python -m src.baselines
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict

from src import config
from src.reliability import aurc, batch_features, ece, load_eval_data, make_pipeline_model, risk_coverage

SIGNALS = {"top1": 0, "mean_topk": 1, "gap12": 2, "vote_share": 3, "agree_frac": 4, "entropy": 5}


def class_probs(scores, cats, alpha_smooth: float = 0.02) -> dict[str, float]:
    mass: dict[str, float] = {}
    for c, s in zip(cats, np.clip(np.asarray(scores), 0, None)):
        mass[c] = mass.get(c, 0.0) + float(s)
    tot = sum(mass.values()) or 1.0
    return {c: m / tot for c, m in mass.items()}


def conformal_sets(cal_scores, cal_cats, cal_gold, test_scores, test_cats, alpha: float):
    """Split conformal with the LAC score 1 - p(true class); sets are {c : p(c) >= 1 - qhat}."""
    nc = np.array([1 - class_probs(s, c).get(g, 0.0) for s, c, g in zip(cal_scores, cal_cats, cal_gold)])
    n = len(nc)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    qhat = float(np.quantile(nc, level, method="higher"))
    return [{c for c, p in class_probs(s, cs).items() if p >= 1 - qhat} for s, cs in zip(test_scores, test_cats)]


def op_at(conf, ok, cov_target):
    rc = risk_coverage(conf, ok)
    row = rc.iloc[min(len(rc) - 1, int(np.ceil(cov_target * len(rc))) - 1)]
    return float(row["risk"])


def cov_at_risk(conf, ok, risk_target):
    rc = risk_coverage(conf, ok)
    good = rc[rc["risk"] <= risk_target]
    return float(good["coverage"].iloc[-1]) if len(good) else 0.0


def main() -> None:
    ev = load_eval_data(4000, config.TOP_K)
    cal_pred, Xc = batch_features(ev["cs"], ev["cc"])
    hold_pred, Xh = batch_features(ev["hs"], ev["hc"])
    yc = np.array([p == g for p, g in zip(cal_pred, ev["cal_gold"])])
    yh = np.array([p == g for p, g in zip(hold_pred, ev["test"]["category"])])
    src_h = ev["test"]["source"].to_numpy()

    methods: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, i in SIGNALS.items():
        sign = -1.0 if name == "entropy" else 1.0  # higher entropy = less confident
        methods[f"single: {name}"] = (sign * Xc[:, i], sign * Xh[:, i])
    lr = make_pipeline_model()
    oof_lr = cross_val_predict(lr, Xc, yc, cv=5, method="predict_proba")[:, 1]
    methods["logistic (deployed)"] = (oof_lr, make_pipeline_model().fit(Xc, yc).predict_proba(Xh)[:, 1])
    gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=150, random_state=config.SEED)
    oof_gb = cross_val_predict(gb, Xc, yc, cv=5, method="predict_proba")[:, 1]
    methods["gradient boosting"] = (oof_gb, gb.fit(Xc, yc).predict_proba(Xh)[:, 1])
    iso = CalibratedClassifierCV(make_pipeline_model(), method="isotonic", cv=5)
    oof_iso = cross_val_predict(iso, Xc, yc, cv=5, method="predict_proba")[:, 1]
    methods["logistic + isotonic"] = (oof_iso, iso.fit(Xc, yc).predict_proba(Xh)[:, 1])

    table = {}
    for name, (c, h) in methods.items():
        table[name] = {
            "aurc_cal": round(aurc(c, yc), 4), "aurc_hold": round(aurc(h, yh), 4),
            "risk_at_80cov_cal": round(op_at(c, yc, 0.8), 4), "risk_at_80cov_hold": round(op_at(h, yh, 0.8), 4),
            "cov_at_10risk_cal": round(cov_at_risk(c, yc, 0.10), 4), "cov_at_10risk_hold": round(cov_at_risk(h, yh, 0.10), 4),
        }
    for name in ("logistic (deployed)", "gradient boosting", "logistic + isotonic"):
        c, h = methods[name]
        table[name]["ece_cal"], table[name]["ece_hold"] = round(ece(c, yc), 4), round(ece(h, yh), 4)

    # split conformal: calibrate on half of the pool, evaluate on the other half and on the held-out split
    rng = np.random.default_rng(config.SEED)
    idx = rng.permutation(len(yc)); a, b = idx[: len(idx) // 2], idx[len(idx) // 2:]
    cal_s, cal_c, cal_g = ev["cs"][a], [ev["cc"][i] for i in a], [ev["cal_gold"][i] for i in a]
    conf_pts = []
    for alpha in [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]:
        sets = conformal_sets(cal_s, cal_c, cal_g, ev["hs"], ev["hc"], alpha)
        single = np.array([len(s) == 1 for s in sets])
        covered = np.array([g in s for s, g in zip(sets, ev["test"]["category"])])
        top = np.array([next(iter(s)) if len(s) == 1 else None for s in sets])
        correct_single = np.array([t == g for t, g in zip(top, ev["test"]["category"])])
        conf_pts.append({
            "alpha": alpha, "coverage_answered": round(float(single.mean()), 4),
            "risk_on_answered": round(float(1 - correct_single[single].mean()), 4) if single.any() else None,
            "set_contains_truth": round(float(covered.mean()), 4),
            "mean_set_size": round(float(np.mean([len(s) for s in sets])), 3),
        })

    # same-coverage comparison against the deployed logistic gate, at each conformal operating point
    for p in conf_pts:
        p["deployed_risk_at_same_coverage"] = round(op_at(*(methods["logistic (deployed)"][1], yh), p["coverage_answered"]), 4) if p["coverage_answered"] > 0 else None
        p["vote_share_risk_at_same_coverage"] = round(op_at(methods["single: vote_share"][1], yh, p["coverage_answered"]), 4) if p["coverage_answered"] > 0 else None

    out = {"n_cal": int(len(yc)), "n_hold": int(len(yh)), "methods": table, "conformal": conf_pts}
    (config.RESULTS_DIR / "baselines.json").write_text(json.dumps(out, indent=2))
    _plot(methods, yh, conf_pts)
    _report(out)


def _plot(methods, yh, conf_pts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for name, color, ls in [("logistic (deployed)", "#2347d0", "-"), ("gradient boosting", "#0c7a5c", "-"),
                            ("single: vote_share", "#7a8493", "--"), ("single: top1", "#b45309", ":")]:
        rc = risk_coverage(methods[name][1], yh)
        ax.plot(rc["coverage"], rc["risk"], ls, color=color, lw=2, label=name)
    pts = [p for p in conf_pts if p["risk_on_answered"] is not None]
    ax.scatter([p["coverage_answered"] for p in pts], [p["risk_on_answered"] for p in pts], marker="D", color="#b42318", zorder=5, label="split conformal (singleton sets)")
    ax.set(xlabel="coverage", ylabel="error on answered", title=f"Held-out (n={len(yh)}): gate vs baselines", ylim=(0, 0.25))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "baselines.png", dpi=140)


def _report(o):
    m = o["methods"]
    lines = ["# Baselines: does the learned gate earn its place?", "",
             f"Same {o['n_cal']:,}-query leave-one-out pool and {o['n_hold']} held-out queries as `reliability.md`. Lower AURC and lower risk are better; higher coverage is better.", "",
             "| Method | AURC (cal) | AURC (held-out) | Error @80% cov (held-out) | Coverage @10% err (cal) | Coverage @10% err (held-out) |", "|---|---|---|---|---|---|"]
    for k, v in sorted(m.items(), key=lambda kv: kv[1]["aurc_cal"]):
        lines.append(f"| {k} | {v['aurc_cal']:.4f} | {v['aurc_hold']:.4f} | {v['risk_at_80cov_hold']:.1%} | {v['cov_at_10risk_cal']:.1%} | {v['cov_at_10risk_hold']:.1%} |")
    lines += ["", "Calibration error (ECE): " + ", ".join(f"{k} {v['ece_cal']:.3f} (cal) / {v['ece_hold']:.3f} (held-out)" for k, v in m.items() if "ece_cal" in v), "",
              "## Split conformal prediction", "",
              "Abstain unless the conformal set holds exactly one category. `contains truth` should be at least 1 - alpha if the guarantee holds.", "",
              "| alpha | answered | error on answered | set contains truth | mean set size | deployed gate error at same coverage | vote-share error at same coverage |", "|---|---|---|---|---|---|---|"]
    for p in o["conformal"]:
        f = lambda x: "-" if x is None else f"{x:.1%}"
        lines.append(f"| {p['alpha']} | {p['coverage_answered']:.1%} | {f(p['risk_on_answered'])} | {p['set_contains_truth']:.1%} | {p['mean_set_size']:.2f} | {f(p['deployed_risk_at_same_coverage'])} | {f(p['vote_share_risk_at_same_coverage'])} |")
    lines += ["", "![baselines](baselines.png)"]
    (config.RESULTS_DIR / "baselines.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
