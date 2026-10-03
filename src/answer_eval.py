"""Evaluate the written recommendations themselves, with and without the confidence gate.

Everything here is objective (no LLM judge needed), because every held-out ticket has a gold
intent and a gold human resolution:

  intent_correct / category_correct   vs the held-out record's gold labels
  resolution_sim      cosine(embedding of the recommendation, embedding of the gold human resolution)
  retrieval_only_sim  same, but for the top-1 retrieved ticket's resolution (a no-LLM baseline)
  support_sim         max cosine between the recommendation and any retrieved resolution
                      (how much the answer stays inside its sources)
  unsupported_specifics   numbers / URLs / timeframes in the answer that appear in neither the
                      retrieved sources nor the customer message (a cheap hallucination proxy)

The gate is applied afterwards: answers whose calibrated confidence is below tau are counted as
escalations. Their quality is reported separately, so you can see what the gate held back.

Usage:
    python -m src.answer_eval --variants grounded --n 200     # runs the LLM, resumable
    python -m src.answer_eval --variants naive --n 100
    python -m src.answer_eval --summary-only                  # rebuild report from saved runs
"""
from __future__ import annotations

import argparse
import json
import re
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from src import config
from src.llm import get_embeddings
from src.rag import SupportRAG
from src.reliability import ConfidenceModel

SPECIFIC = re.compile(
    r"https?://\S+|www\.\S+|\b\d+(?:[.,:]\d+)*\s*(?:-\s*\d+\s*)?(?:%|hours?|hrs?|days?|business days|minutes?|mins?|weeks?|months?)?",
    re.I,
)


def norm(s) -> str:
    return str(s).strip().lower().replace(" ", "_")


def specifics(text: str) -> set[str]:
    return {m.group(0).strip().lower().rstrip(".,") for m in SPECIFIC.finditer(text) if re.search(r"\d|http|www", m.group(0))}


def unsupported(answer: str, allowed: str) -> tuple[int, int]:
    s = specifics(answer)
    allowed_l = allowed.lower()
    bad = [x for x in s if x not in allowed_l and re.sub(r"\s*(hours?|days?|minutes?|weeks?|months?|%)\b.*", "", x) not in allowed_l]
    return len(s), len(bad)


def path_for(variant: str):
    return config.RESULTS_DIR / f"answer_eval_{variant}.jsonl"


def run_variant(rag: SupportRAG, gate: ConfidenceModel, test: pd.DataFrame, variant: str) -> None:
    path = path_for(variant)
    done = {json.loads(l)["ticket_id"] for l in path.read_text().splitlines()} if path.exists() else set()
    emb = get_embeddings()
    with path.open("a") as f:
        for _, row in tqdm(test[~test["ticket_id"].isin(done)].iterrows(), total=len(test) - len(done), desc=variant):
            t0 = time.time()
            try:
                res = rag.run(row["query"], variant)
            except Exception as e:  # small local models occasionally emit invalid JSON; skip, rerun resumes
                tqdm.write(f"  ! {row['ticket_id']}: {type(e).__name__}")
                continue
            latency = time.time() - t0
            full = rag.store.similarity_search_with_score(row["query"], k=rag.top_k)
            _, conf = gate.confidence([s for _, s in full], [d.metadata["category"] for d, _ in full])
            ins = res.insight
            answer = " ".join([ins["next_best_action"]] + ins["steps"])
            vecs = np.asarray(emb.embed_documents([answer, row["resolution"], full[0][0].metadata["resolution"]]))
            sources_text = row["query"] + " " + " ".join(res.contexts)
            n_spec, n_bad = unsupported(answer, sources_text)
            ctx_vecs = np.asarray(emb.embed_documents([d.metadata["resolution"] for d, _ in full]))
            rec = {
                "ticket_id": row["ticket_id"], "variant": variant, "source": row["source"],
                "query": row["query"], "gold_category": row["category"], "gold_intent": row["intent"],
                "gold_resolution": row["resolution"][:600],
                "pred_category": ins["category"], "pred_intent": ins["intent"], "answer": answer,
                "cited": ins["cited_sources"], "llm_escalate": ins["escalate"], "llm_confidence": ins["confidence"],
                "gate_conf": conf, "gate_abstain": bool(conf < gate.tau),
                "intent_correct": norm(ins["intent"]) == norm(row["intent"]),
                "category_correct": norm(ins["category"]) == norm(row["category"]),
                "resolution_sim": float(vecs[0] @ vecs[1]),
                "retrieval_only_sim": float(vecs[2] @ vecs[1]),
                "support_sim": float((ctx_vecs @ vecs[0]).max()),
                "n_specifics": n_spec, "n_unsupported": n_bad, "has_unsupported": n_bad > 0,
                "latency_s": round(latency, 2),
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()


_GATE_CACHE: dict = {}


def apply_current_gate(df: pd.DataFrame) -> pd.DataFrame:
    """Gate flags are recomputed from the current gate on every summary, so refitting the gate never
    leaves stale decisions in the saved generations."""
    from src.reliability import batch_features, load_eval_data

    if "feat" not in _GATE_CACHE:
        ev = load_eval_data(4000, config.TOP_K)
        _, Xh = batch_features(ev["hs"], ev["hc"])
        _GATE_CACHE["feat"] = dict(zip(ev["test"]["ticket_id"], Xh))
        _GATE_CACHE["gate"] = ConfidenceModel.load()
    gate = _GATE_CACHE["gate"]
    X = np.vstack([_GATE_CACHE["feat"][t] for t in df["ticket_id"]])
    conf = gate.predict_proba(X)
    ood = (X[:, 0] < gate.ood_top1) if gate.ood_top1 is not None else np.zeros(len(conf), bool)
    out = df.copy()
    out["gate_conf"], out["gate_abstain"] = conf, (conf < gate.tau) | ood
    return out


def load(variant: str) -> pd.DataFrame | None:
    p = path_for(variant)
    return apply_current_gate(pd.read_json(p, lines=True)) if p.exists() else None


def boot_ci(x: np.ndarray, n: int = 1000, seed: int = config.SEED) -> list[float]:
    rng = np.random.default_rng(seed)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def block(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {"n": 0}
    return {
        "n": int(len(df)),
        "category_acc": round(float(df["category_correct"].mean()), 4),
        "intent_acc": round(float(df["intent_correct"].mean()), 4),
        "intent_acc_ci95": boot_ci(df["intent_correct"].to_numpy(float)),
        "resolution_sim": round(float(df["resolution_sim"].mean()), 4),
        "retrieval_only_sim": round(float(df["retrieval_only_sim"].mean()), 4),
        "support_sim": round(float(df["support_sim"].mean()), 4),
        "unsupported_specifics_rate": round(float(df["has_unsupported"].mean()), 4),
        "cites_a_source": round(float((df["cited"].map(len) > 0).mean()), 4),
        "p50_latency_s": round(float(df["latency_s"].median()), 2),
    }


def summarise(judge: dict | None = None, human: dict | None = None) -> dict:
    out: dict = {"variants": {}}
    g = load("grounded")
    for v in ("naive", "grounded"):
        d = load(v)
        if d is None:
            continue
        out["variants"][v] = {"all": block(d)}
        if v == "grounded":
            out["variants"][v]["gate_answered"] = block(d[~d["gate_abstain"]])
            out["variants"][v]["gate_escalated"] = block(d[d["gate_abstain"]])
            out["variants"][v]["by_source_all"] = {s: block(x) for s, x in d.groupby("source")}
            out["variants"][v]["by_source_answered"] = {s: block(x) for s, x in d[~d["gate_abstain"]].groupby("source")}
    if g is not None and "naive" in out["variants"]:
        nv = load("naive")
        common = g[g["ticket_id"].isin(nv["ticket_id"])]
        if len(common) < 30:  # too few paired queries to say anything; leave the comparison out
            return {k: v for k, v in out.items() if k != "same_queries_grounded_vs_naive"} | {"llm_selfescalation_vs_gate": {
                "llm_escalated_itself": round(float(g["llm_escalate"].mean()), 4), "gate_escalated": round(float(g["gate_abstain"].mean()), 4)}}
        out["same_queries_grounded_vs_naive"] = {
            "n": int(len(common)),
            "grounded": block(common),
            "naive": block(nv[nv["ticket_id"].isin(common["ticket_id"])]),
            "grounded_gated_answered": block(common[~common["gate_abstain"]]),
        }
    if g is not None:
        out["llm_selfescalation_vs_gate"] = {
            "llm_escalated_itself": round(float(g["llm_escalate"].mean()), 4),
            "gate_escalated": round(float(g["gate_abstain"].mean()), 4),
        }
    if judge:
        out["llm_judge_sample"] = judge
    if human:
        out["human_grades"] = human
    return out


def write_report(s: dict, path) -> None:
    v = s["variants"].get("grounded")
    rows = [("Intent accuracy", "intent_acc", True), ("Category accuracy", "category_acc", True),
            ("Similarity to the gold human resolution", "resolution_sim", False),
            ("(baseline) top-1 retrieved resolution", "retrieval_only_sim", False),
            ("Stays inside its sources (support sim)", "support_sim", False),
            ("Answers with unsupported numbers/timeframes", "unsupported_specifics_rate", True),
            ("Cites at least one source", "cites_a_source", True)]
    lines = ["# Answer-quality evaluation (objective metrics, gold labels, no judge)", ""]
    if v:
        cols = [("all", "Ungated (answer everything)"), ("gate_answered", "Gate: answered"), ("gate_escalated", "Gate: escalated")]
        lines += [f"Grounded pipeline on {v['all']['n']} held-out queries.", "",
                  "| Metric | " + " | ".join(c[1] for c in cols) + " |", "|---|" + "---|" * len(cols),
                  "| n | " + " | ".join(str(v[k]["n"]) for k, _ in cols) + " |"]
        for lab, key, pct in rows:
            lines.append(f"| {lab} | " + " | ".join((f"{v[k][key]:.1%}" if pct else f"{v[k][key]:.3f}") if v[k]["n"] else "-" for k, _ in cols) + " |")
        lines += ["", "Intent accuracy 95% CI (ungated): " + str(v["all"]["intent_acc_ci95"]) + ", (gate answered): " + str(v["gate_answered"].get("intent_acc_ci95")), ""]
    c = s.get("same_queries_grounded_vs_naive")
    if c:
        lines += [f"## Grounded vs naive prompt on the same {c['n']} queries", "", "| Metric | naive | grounded | grounded + gate (answered) |", "|---|---|---|---|"]
        for lab, key, pct in rows:
            f = (lambda x: f"{x:.1%}") if pct else (lambda x: f"{x:.3f}")
            lines.append(f"| {lab} | {f(c['naive'][key])} | {f(c['grounded'][key])} | {f(c['grounded_gated_answered'][key]) if c['grounded_gated_answered']['n'] else '-'} |")
        lines.append("")
    if s.get("llm_judge_sample"):
        j = s["llm_judge_sample"]
        lines += ["## Stronger-model judge on a sample", "", f"Judge `{j['judge_model']}` on {j['n']} grounded answers: mean faithfulness {j['mean_faithfulness']:.2f}, "
                  f"hallucination rate {j['hallucination_rate']:.1%} (answered by the gate: {j['hallucination_rate_gate_answered']:.1%}, escalated: {j['hallucination_rate_gate_escalated']:.1%}).", ""]
    if s.get("human_grades"):
        h = s["human_grades"]
        lines += ["## Human grades", "", "| group | n | mean action score (0-2) | rated good | grounded |", "|---|---|---|---|---|"]
        lines += [f"| {k} | {v['n']} | {v['mean_action_ok_0to2']} | {v['good_rate']:.0%} | {v['grounded_rate']} |" for k, v in h.items()] + [""]
    path.write_text("\n".join(lines) + "\n")


HUMAN_SHEET = config.RESULTS_DIR / "human_grading_sheet.csv"


def make_human_sheet(n: int) -> None:
    """Blank grading sheet for a person to fill in. Columns to fill: action_ok (0/1/2) and grounded (0/1)."""
    g = load("grounded")
    s = g.sample(min(n, len(g)), random_state=config.SEED + 1)
    sheet = pd.DataFrame({
        "ticket_id": s["ticket_id"], "customer_message": s["query"].str.slice(0, 500),
        "gold_human_resolution": s["gold_resolution"].str.slice(0, 500), "system_recommendation": s["answer"],
        "gate_decision": s["gate_abstain"].map({True: "escalated", False: "answered"}),
        "action_ok": "", "grounded": "", "notes": "",
    })
    sheet.to_csv(HUMAN_SHEET, index=False)
    print(f"wrote {HUMAN_SHEET}: grade action_ok (0 wrong, 1 partly useful, 2 good) and grounded (0 invents details, 1 stays in sources)")


def human_summary(path: str) -> dict | None:
    d = pd.read_csv(path)
    d = d[d["action_ok"].astype(str).str.strip() != ""].copy()
    if d.empty:
        return None
    d["action_ok"] = d["action_ok"].astype(float)
    d["grounded"] = pd.to_numeric(d["grounded"], errors="coerce")
    by = lambda x: {"n": int(len(x)), "mean_action_ok_0to2": round(float(x["action_ok"].mean()), 3), "good_rate": round(float((x["action_ok"] == 2).mean()), 3),
                    "grounded_rate": round(float(x["grounded"].mean()), 3) if x["grounded"].notna().any() else None}
    return {"all": by(d), **{k: by(g) for k, g in d.groupby("gate_decision")}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=[], choices=["naive", "grounded"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--make-human-sheet", type=int, metavar="N", help="write a blank grading sheet with N sampled answers")
    ap.add_argument("--human", metavar="CSV", help="fold a completed grading sheet into the report")
    args = ap.parse_args()
    if args.make_human_sheet:
        return make_human_sheet(args.make_human_sheet)

    if args.variants and not args.summary_only:
        gate = ConfidenceModel.load()
        assert gate is not None, "run `python -m src.reliability` first"
        rag = SupportRAG(gate=None)
        test = pd.read_csv(config.TEST_CSV).head(args.n)
        for v in args.variants:
            run_variant(rag, gate, test, v)

    jp = config.RESULTS_DIR / "judge_sample.json"
    s = summarise(judge=json.loads(jp.read_text()) if jp.exists() else None, human=human_summary(args.human) if args.human else None)
    (config.RESULTS_DIR / "answer_eval.json").write_text(json.dumps(s, indent=2))
    write_report(s, config.RESULTS_DIR / "answer_eval.md")
    print((config.RESULTS_DIR / "answer_eval.md").read_text())


if __name__ == "__main__":
    main()
