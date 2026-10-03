"""Export everything the static demo page needs to results/demo_data.json.

The demo is a single self-contained HTML file with no backend, so it can be hosted anywhere
(GitHub Pages, Hugging Face Spaces, an artifact link). All numbers on it come from this export.

Usage:  python -m src.export_demo            (runs ~10 real LLM calls for the showcase answers)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config
from src.ingest import load_index
from src.rag import SupportRAG
from src.reliability import ConfidenceModel, _docs, batch_features, load_eval_data, search_vectors

OUT = config.RESULTS_DIR / "demo_data.json"
SHOWCASE_CACHE = config.RESULTS_DIR / "demo_showcase.json"


def clip(s: str, n: int) -> str:
    s = " ".join(str(s).replace("\\n", " ").replace("<br>", " ").split())
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def pick_showcase(df: pd.DataFrame) -> list[str]:
    """A fixed, representative mix: easy answered, hard answered, abstained, and one confident miss."""
    ids: list[str] = []
    answered = df[~df["abstain"]].sort_values("conf", ascending=False)
    ids += answered[(answered["src"] == "bitext") & answered["ok"]].head(2)["id"].tolist()
    ids += answered[(answered["src"] == "tickets") & answered["ok"]].head(3)["id"].tolist()
    miss = answered[~answered["ok"]]
    ids += miss.head(1)["id"].tolist()
    ids += df[df["abstain"] & (df["src"] == "tickets")].sort_values("conf").head(3)["id"].tolist()
    return ids


def main() -> None:
    gate = ConfidenceModel.load()
    assert gate is not None, "run `python -m src.reliability` first"
    store = load_index()
    docs = _docs(store)
    test = pd.read_csv(config.TEST_CSV)

    vecs = load_eval_data(4000, config.TOP_K)["hold_vecs"]  # cached embeddings: no LLM needed
    scores, cats = search_vectors(store, vecs, config.TOP_K)
    pred, X = batch_features(scores, cats)
    conf = gate.predict_proba(X)
    reasons = [gate.decide(sc, ct)[2] for sc, ct in zip(scores, cats)]
    # neighbour details (search_vectors drops them, so look up again)
    raw_scores, raw_idx = store.index.search(vecs.astype(np.float32), config.TOP_K)

    rows = []
    for i, r in test.reset_index(drop=True).iterrows():
        nn = []
        for sc, ix in list(zip(raw_scores[i], raw_idx[i]))[:3]:
            m = docs[ix].metadata
            nn.append({"cat": m["category"], "sim": round(float(sc), 3), "q": clip(m["query"], 170), "res": clip(m["resolution"], 230)})
        rows.append(
            {
                "id": r["ticket_id"],
                "q": clip(r["query"], 420),
                "gold": r["category"],
                "src": r["source"],
                "pred": pred[i],
                "conf": round(float(conf[i]), 4),
                "ok": bool(pred[i] == r["category"]),
                "abstain": reasons[i] is not None,
                "reason": reasons[i],
                "nn": nn,
            }
        )
    df = pd.DataFrame(rows)

    if SHOWCASE_CACHE.exists():
        showcase = json.loads(SHOWCASE_CACHE.read_text())
    else:
        rag = SupportRAG(gate=gate)
        showcase = {}
        for tid in pick_showcase(df):
            row = test[test["ticket_id"] == tid].iloc[0]
            res = rag.run(row["query"], "grounded")
            ins = res.insight
            showcase[tid] = {
                "category": ins["category"],
                "intent": ins["intent"],
                "action": clip(ins["next_best_action"], 300),
                "steps": [clip(s, 200) for s in ins["steps"][:4]],
                "escalate": ins["escalate"],
                "abstained": res.abstained,
                "reason": res.abstain_reason,
            }
            print("showcase", tid, res.abstained)
        SHOWCASE_CACHE.write_text(json.dumps(showcase, indent=1))

    rel = json.loads((config.RESULTS_DIR / "reliability.json").read_text())
    study_path = config.RESULTS_DIR / "confidence_study.json"
    def load_json(name):
        p = config.RESULTS_DIR / name
        return json.loads(p.read_text()) if p.exists() else None

    data = {
        "meta": {k: rel[k] for k in ("n_calibration", "n_holdout", "index_size", "k", "target_risk", "tau",
                                      "baseline_accuracy", "aurc", "ece", "at_threshold", "by_source",
                                      "within_source_aurc", "model", "importance", "ood_top1")},
        "gaps": rel["knowledge_gaps"],
        "cal": json.loads((config.RESULTS_DIR / "cal_points.json").read_text()),
        "holdout": rows,
        "showcase": showcase,
        "study": json.loads(study_path.read_text()) if study_path.exists() else None,
        "baselines": load_json("baselines.json"),
        "stress": load_json("stress_test.json"),
        "stress_aug": load_json("stress_test_aug.json"),
        "stress_before": load_json("stress_test_before.json"),
        "failure": load_json("failure_analysis.json"),
        "answer": load_json("answer_eval.json"),
        "bench": load_json("benchmark.json"),
        "models": {"chat": config.CHAT_MODEL, "embed": config.EMBED_MODEL},
    }
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB), showcase={len(showcase)}")


if __name__ == "__main__":
    main()
