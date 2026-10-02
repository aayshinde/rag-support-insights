"""Reliability layer: know when the pipeline should NOT answer.

A RAG triage system is only useful if it can tell a trustworthy answer from a guess. This
module turns retrieval evidence into a *calibrated probability that the triage is correct*,
gates answers on it (abstain -> escalate to a human), and measures the trade-off.

Signals (all from the retrieved neighbours, no extra LLM call)
    top1, mean_topk, gap12    raw similarity strength
    vote_share                share of similarity mass held by the winning category
    agree_frac                fraction of the k neighbours that share the winning category
    entropy                   uncertainty of the similarity-weighted category distribution

Label: is the neighbour-vote category equal to the gold category? Objective, so it needs no
LLM judge and can be measured on thousands of queries in seconds.

Honest evaluation: the confidence model is fit on *leave-one-out* queries drawn from the
knowledge base (each query is searched with itself removed), and then tested on the
held-out split, which was never indexed.

Usage:
    python -m src.reliability                 # fit, evaluate, write results/ + index/confidence.joblib
    python -m src.reliability --target-risk 0.05
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.ingest import load_index
from src.llm import get_embeddings

FEATURES = ["top1", "mean_topk", "gap12", "vote_share", "agree_frac", "entropy"]
MODEL_PATH = config.INDEX_DIR / "confidence.joblib"


# ---------------------------------------------------------------- features


def vote_features(scores, cats: list[str]) -> tuple[str, np.ndarray]:
    """Return (neighbour-vote category, feature vector) for one query's retrieved neighbours."""
    s = np.asarray(scores, dtype=np.float64)
    w = np.clip(s, 0.0, None)
    mass: dict[str, float] = {}
    for c, wi in zip(cats, w):
        mass[c] = mass.get(c, 0.0) + wi
    total = sum(mass.values())
    if total <= 0:  # degenerate: fall back to unweighted counts
        mass = {c: float(cats.count(c)) for c in set(cats)}
        total = sum(mass.values())
    winner = max(mass, key=mass.get)
    p = np.array([m / total for m in mass.values()])
    feats = np.array(
        [
            s[0],
            s.mean(),
            s[0] - s[1] if len(s) > 1 else 0.0,
            mass[winner] / total,
            cats.count(winner) / len(cats),
            float(-(p * np.log(p + 1e-12)).sum()),
        ]
    )
    return winner, feats


def batch_features(score_rows: np.ndarray, cat_rows: list[list[str]]) -> tuple[list[str], np.ndarray]:
    out = [vote_features(s, c) for s, c in zip(score_rows, cat_rows)]
    return [w for w, _ in out], np.vstack([f for _, f in out])


# ---------------------------------------------------------------- retrieval over the index


def _docs(store) -> list:
    return [store.docstore.search(store.index_to_docstore_id[i]) for i in range(store.index.ntotal)]


def search_vectors(store, vecs: np.ndarray, k: int, exclude_ids: list[str] | None = None):
    """Batch nearest-neighbour search. If exclude_ids is given, drop the neighbour whose ticket_id
    matches (leave-one-out) and any neighbour with identical query text."""
    docs = _docs(store)
    by_id = {d.metadata["ticket_id"]: d for d in docs}
    extra = 3 if exclude_ids is not None else 0
    scores, idx = store.index.search(vecs.astype(np.float32), k + extra)
    score_rows, cat_rows = [], []
    for qi, (srow, irow) in enumerate(zip(scores, idx)):
        keep_s, keep_c = [], []
        own = by_id.get(exclude_ids[qi]) if exclude_ids is not None else None
        for sc, ix in zip(srow, irow):
            if ix < 0:
                continue
            d = docs[ix]
            if exclude_ids is not None and (d.metadata["ticket_id"] == exclude_ids[qi] or (
                own is not None and d.metadata["query"] == own.metadata["query"]
            )):
                continue
            keep_s.append(float(sc))
            keep_c.append(d.metadata["category"])
            if len(keep_s) == k:
                break
        score_rows.append(keep_s)
        cat_rows.append(keep_c)
    return np.array(score_rows), cat_rows


def embed_queries(texts: list[str]) -> np.ndarray:
    return np.asarray(get_embeddings().embed_documents(texts), dtype=np.float32)


# ---------------------------------------------------------------- confidence model


@dataclass
class ConfidenceModel:
    pipeline: object
    tau: float
    target_risk: float

    def predict_proba(self, feats: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(np.atleast_2d(feats))[:, 1]

    def confidence(self, scores, cats: list[str]) -> tuple[str, float]:
        winner, f = vote_features(scores, cats)
        return winner, float(self.predict_proba(f)[0])

    def save(self, path=MODEL_PATH) -> None:
        joblib.dump({"pipeline": self.pipeline, "tau": self.tau, "target_risk": self.target_risk, "features": FEATURES}, path)

    @classmethod
    def load(cls, path=MODEL_PATH) -> "ConfidenceModel | None":
        if not path.exists():
            return None
        d = joblib.load(path)
        return cls(d["pipeline"], d["tau"], d["target_risk"])


def make_pipeline_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))


# ---------------------------------------------------------------- selective-prediction metrics


def risk_coverage(conf: np.ndarray, correct: np.ndarray) -> pd.DataFrame:
    """Answer the most-confident queries first; report error rate (risk) at each coverage."""
    order = np.argsort(-conf, kind="stable")
    c = correct[order].astype(float)
    n = len(c)
    cov = np.arange(1, n + 1) / n
    risk = 1 - np.cumsum(c) / np.arange(1, n + 1)
    return pd.DataFrame({"coverage": cov, "risk": risk, "conf": conf[order]})


def aurc(conf, correct) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    return float(risk_coverage(conf, correct)["risk"].mean())


def ece(conf, correct, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def pick_tau(conf: np.ndarray, correct: np.ndarray, target_risk: float) -> float:
    """Lowest threshold whose accepted set has error <= target_risk (max coverage at that risk)."""
    rc = risk_coverage(conf, correct)
    ok = rc[rc["risk"] <= target_risk]
    return float(ok["conf"].iloc[-1]) if len(ok) else float(conf.max() + 1e-6)


def at_threshold(conf, correct, tau: float, n_boot: int = 1000, seed: int = config.SEED) -> dict:
    acc = conf >= tau
    out = {"coverage": float(acc.mean()), "risk": float(1 - correct[acc].mean()) if acc.any() else float("nan")}
    rng = np.random.default_rng(seed)
    risks = []
    for _ in range(n_boot):
        i = rng.integers(0, len(conf), len(conf))
        a = conf[i] >= tau
        if a.any():
            risks.append(1 - correct[i][a].mean())
    out["risk_ci95"] = [float(np.percentile(risks, 2.5)), float(np.percentile(risks, 97.5))] if risks else None
    return out


def by_source(sources, conf, correct, tau: float) -> dict:
    """The data mixes templated (easy) and free-text (hard) tickets; pooled numbers hide that."""
    out = {}
    sources = np.asarray(sources)
    for src in sorted(set(sources)):
        m = sources == src
        acc = m & (conf >= tau)
        out[src] = {
            "n": int(m.sum()),
            "baseline_accuracy": round(float(correct[m].mean()), 4),
            "coverage": round(float(acc.sum() / m.sum()), 4),
            "error_on_answered": round(float(1 - correct[acc].mean()), 4) if acc.any() else None,
        }
    return out


# ---------------------------------------------------------------- knowledge gaps


def find_gaps(texts: list[str], vecs: np.ndarray, conf: np.ndarray, gold: list[str], tau: float, max_clusters: int = 8):
    """Cluster the queries the system would abstain on, and label each cluster with TF-IDF terms."""
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    low = np.where(conf < tau)[0]
    if len(low) < 10:
        return []
    k = int(min(max_clusters, max(2, len(low) // 15)))
    km = KMeans(n_clusters=k, n_init=10, random_state=config.SEED).fit(vecs[low])
    tf = TfidfVectorizer(stop_words="english", max_features=4000, ngram_range=(1, 2), min_df=2)
    X = tf.fit_transform([texts[i] for i in low])
    terms = np.array(tf.get_feature_names_out())
    gaps = []
    for c in range(k):
        members = low[km.labels_ == c]
        local = np.where(km.labels_ == c)[0]
        top = terms[np.asarray(X[local].mean(axis=0)).ravel().argsort()[::-1][:5]].tolist()
        center = km.cluster_centers_[c]
        near = members[np.argsort(np.linalg.norm(vecs[members] - center, axis=1))[:3]]
        cats = pd.Series([gold[i] for i in members]).value_counts()
        gaps.append(
            {
                "size": int(len(members)),
                "share_of_abstained": round(len(members) / len(low), 3),
                "keywords": top,
                "mean_confidence": round(float(conf[members].mean()), 3),
                "top_gold_category": cats.index[0],
                "category_purity": round(float(cats.iloc[0] / len(members)), 3),
                "examples": [texts[i][:240] for i in near],
            }
        )
    return sorted(gaps, key=lambda g: -g["size"])


# ---------------------------------------------------------------- main


def _plots(path, cal, hold, tau):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for name, (conf, correct), style in [
        ("calibrated model", cal["model"], "-"),
        ("raw top-1 similarity", cal["top1"], "--"),
        ("vote share", cal["vote"], ":"),
    ]:
        rc = risk_coverage(conf, correct)
        ax[0].plot(rc["coverage"], rc["risk"], style, label=name, lw=2)
    ax[0].axhline(1 - cal["model"][1].mean(), color="gray", lw=1, label="answer everything")
    ax[0].set(xlabel="coverage (share of queries answered)", ylabel="error rate on answered", title="Risk-coverage (calibration, out-of-fold)")
    ax[0].legend()

    rc = risk_coverage(*hold["model"])
    ax[1].plot(rc["coverage"], rc["risk"], lw=2, label="calibrated model (held-out)")
    ax[1].axhline(1 - hold["model"][1].mean(), color="gray", lw=1, label="answer everything")
    t = hold["at_tau"]
    ax[1].scatter([t["coverage"]], [t["risk"]], color="red", zorder=5, label=f"chosen threshold ({t['coverage']:.0%} cov, {t['risk']:.1%} err)")
    ax[1].set(xlabel="coverage", ylabel="error rate on answered", title=f"Held-out split (n={len(hold['model'][0])})")
    ax[1].legend()

    conf, correct = cal["model"]
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, 9)
    xs = [conf[idx == b].mean() for b in range(10) if (idx == b).any()]
    ys = [correct[idx == b].mean() for b in range(10) if (idx == b).any()]
    ax[2].plot([0, 1], [0, 1], "k--", lw=1)
    ax[2].plot(xs, ys, "o-", lw=2)
    ax[2].set(xlabel="predicted P(correct)", ylabel="observed accuracy", title="Reliability diagram")
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-cal", type=int, default=2000, help="knowledge-base queries used for leave-one-out calibration")
    ap.add_argument("--target-risk", type=float, default=0.10, help="max error rate allowed on answered queries")
    ap.add_argument("--k", type=int, default=config.TOP_K)
    args = ap.parse_args()

    store = load_index()
    docs = _docs(store)
    rng = np.random.default_rng(config.SEED)
    pick = rng.choice(len(docs), size=min(args.n_cal, len(docs)), replace=False)
    cal_texts = [docs[i].metadata["query"] for i in pick]
    cal_ids = [docs[i].metadata["ticket_id"] for i in pick]
    cal_gold = [docs[i].metadata["category"] for i in pick]
    src_of = pd.read_csv(config.KB_CSV).set_index("ticket_id")["source"].to_dict()
    cal_src = [src_of[t] for t in cal_ids]

    cache = config.RESULTS_DIR / f"cal_emb_{len(docs)}_{len(pick)}_{config.EMBED_MODEL.replace(':', '_')}.npy"
    if cache.exists():
        cal_vecs = np.load(cache)
    else:
        print(f"Embedding {len(pick)} calibration queries ...")
        cal_vecs = embed_queries(cal_texts)
        np.save(cache, cal_vecs)

    test = pd.read_csv(config.TEST_CSV)
    print(f"Embedding {len(test)} held-out queries ...")
    hold_vecs = embed_queries(test["query"].tolist())

    cs, cc = search_vectors(store, cal_vecs, args.k, exclude_ids=cal_ids)
    hs, hc = search_vectors(store, hold_vecs, args.k)
    cal_pred, Xc = batch_features(cs, cc)
    hold_pred, Xh = batch_features(hs, hc)
    yc = np.array([p == g for p, g in zip(cal_pred, cal_gold)])
    yh = np.array([p == g for p, g in zip(hold_pred, test["category"])])

    # out-of-fold confidences for the calibration pool, then the final model on all of it
    oof = cross_val_predict(make_pipeline_model(), Xc, yc, cv=5, method="predict_proba")[:, 1]
    model = make_pipeline_model().fit(Xc, yc)
    tau = pick_tau(oof, yc, args.target_risk)
    cm = ConfidenceModel(model, tau, args.target_risk)
    cm.save()
    hold_conf = cm.predict_proba(Xh)

    signals = {
        "calibrated_model": (oof, hold_conf),
        "raw_top1": (Xc[:, 0], Xh[:, 0]),
        "vote_share": (Xc[:, 3], Xh[:, 3]),
        "random": (rng.random(len(yc)), rng.random(len(yh))),
    }
    summary = {
        "n_calibration": int(len(yc)),
        "n_holdout": int(len(yh)),
        "index_size": len(docs),
        "k": args.k,
        "target_risk": args.target_risk,
        "tau": round(tau, 4),
        "baseline_accuracy": {"calibration": round(float(yc.mean()), 4), "holdout": round(float(yh.mean()), 4)},
        "aurc": {n: {"calibration": round(aurc(c, yc), 4), "holdout": round(aurc(h, yh), 4)} for n, (c, h) in signals.items()},
        "ece": {"calibrated_model": round(ece(oof, yc), 4), "holdout": round(ece(hold_conf, yh), 4)},
        "at_threshold": {"calibration": at_threshold(oof, yc, tau), "holdout": at_threshold(hold_conf, yh, tau)},
        "by_source": {
            "calibration": by_source(cal_src, oof, yc, tau),
            "holdout": by_source(test["source"], hold_conf, yh, tau),
        },
        "within_source_aurc": {
            src: {
                "calibrated_model": round(aurc(oof[np.array(cal_src) == src], yc[np.array(cal_src) == src]), 4),
                "raw_top1": round(aurc(Xc[np.array(cal_src) == src, 0], yc[np.array(cal_src) == src]), 4),
                "vote_share": round(aurc(Xc[np.array(cal_src) == src, 3], yc[np.array(cal_src) == src]), 4),
                "random": round(aurc(rng.random((np.array(cal_src) == src).sum()), yc[np.array(cal_src) == src]), 4),
            }
            for src in sorted(set(cal_src))
        },
        "coefficients": dict(zip(FEATURES, np.round(model[-1].coef_[0], 3).tolist())),
    }

    all_texts = cal_texts + test["query"].tolist()
    all_vecs = np.vstack([cal_vecs, hold_vecs])
    all_conf = np.concatenate([oof, hold_conf])
    all_gold = cal_gold + test["category"].tolist()
    summary["knowledge_gaps"] = find_gaps(all_texts, all_vecs, all_conf, all_gold, tau)

    (config.RESULTS_DIR / "reliability.json").write_text(json.dumps(summary, indent=2))
    (config.RESULTS_DIR / "cal_points.json").write_text(
        json.dumps({"conf": np.round(oof, 4).tolist(), "ok": yc.astype(int).tolist(), "src": cal_src})
    )
    _plots(
        config.RESULTS_DIR / "reliability.png",
        {"model": (oof, yc), "top1": (Xc[:, 0], yc), "vote": (Xc[:, 3], yc)},
        {"model": (hold_conf, yh), "at_tau": summary["at_threshold"]["holdout"]},
        tau,
    )
    write_report(summary, config.RESULTS_DIR / "reliability.md")
    print((config.RESULTS_DIR / "reliability.md").read_text())


def write_report(s: dict, path) -> None:
    h, c = s["at_threshold"]["holdout"], s["at_threshold"]["calibration"]
    base = s["baseline_accuracy"]
    lines = [
        "# Reliability report: when should the system abstain?",
        "",
        f"- Index: **{s['index_size']:,}** records, k={s['k']}. Confidence model fit on **{s['n_calibration']:,}** leave-one-out "
        f"knowledge-base queries; tested on **{s['n_holdout']}** held-out queries never indexed.",
        f"- Target: error on answered queries <= **{s['target_risk']:.0%}** -> abstention threshold tau = **{s['tau']}**",
        "",
        "## Headline",
        "",
        f"Answering everything is correct **{base['holdout']:.1%}** of the time on held-out queries. With the confidence gate, the system "
        f"answers **{h['coverage']:.1%}** of queries at **{h['risk']:.1%}** error (95% CI {h['risk_ci95'][0]:.1%} to {h['risk_ci95'][1]:.1%}) "
        f"and escalates the rest to a human.",
        "",
        "| | coverage | error on answered | baseline error (answer all) |",
        "|---|---|---|---|",
        f"| calibration (out-of-fold) | {c['coverage']:.1%} | {c['risk']:.1%} | {1 - base['calibration']:.1%} |",
        f"| held-out | {h['coverage']:.1%} | {h['risk']:.1%} | {1 - base['holdout']:.1%} |",
        "",
        "## Which signal ranks answers best? (AURC, lower is better)",
        "",
        "| signal | calibration | held-out |",
        "|---|---|---|",
    ]
    for n, v in s["aurc"].items():
        lines.append(f"| {n} | {v['calibration']:.4f} | {v['holdout']:.4f} |")
    lines += [
        "",
        "## Pooled numbers hide a data-mix effect (Simpson's paradox)",
        "",
        "The data mixes short templated tickets (`bitext`, easy) with long free-text tickets (`tickets`, hard). Pooled, raw similarity ranks "
        "answers *worse than random*, because easy short queries score lower similarity than hard long ones. Within each source it is a weak but "
        "positive signal. Per-source behaviour of the gate on the held-out split:",
        "",
        "| source | n | answer-all accuracy | coverage | error on answered |",
        "|---|---|---|---|---|",
        *[
            f"| {k} | {v['n']} | {v['baseline_accuracy']:.1%} | {v['coverage']:.1%} | "
            f"{'n/a' if v['error_on_answered'] is None else format(v['error_on_answered'], '.1%')} |"
            for k, v in s["by_source"]["holdout"].items()
        ],
        "",
        "Within-source AURC on the calibration pool (lower is better):",
        "",
        "| source | calibrated | raw top-1 | vote share | random |",
        "|---|---|---|---|---|",
        *[
            f"| {k} | {v['calibrated_model']:.4f} | {v['raw_top1']:.4f} | {v['vote_share']:.4f} | {v['random']:.4f} |"
            for k, v in s["within_source_aurc"].items()
        ],
        "",
        f"Calibration error (ECE): {s['ece']['calibrated_model']:.3f} out-of-fold, {s['ece']['holdout']:.3f} held-out.",
        f"Standardised coefficients: {s['coefficients']}",
        "",
        "![risk-coverage](reliability.png)",
        "",
        "## Knowledge gaps (clusters of queries the system abstains on)",
        "",
    ]
    for i, g in enumerate(s["knowledge_gaps"], 1):
        lines += [
            f"**{i}. {', '.join(g['keywords'][:3])}** - {g['size']} queries ({g['share_of_abstained']:.0%} of abstentions), "
            f"mean confidence {g['mean_confidence']}, mostly `{g['top_gold_category']}` ({g['category_purity']:.0%})",
            *[f"  - \"{e}\"" for e in g["examples"][:2]],
            "",
        ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
