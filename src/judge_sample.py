"""LLM-judge faithfulness on a sample of generated answers, using a stronger model than the generator.

The generator is a 3B model, so the judge is a larger local model (default qwen2.5:7b-instruct).
Contexts are rebuilt exactly as the pipeline saw them (grounded retrieval + similarity threshold).
Results are cached per answer, so an interrupted run resumes.

Usage:  python -m src.judge_sample --n 60 --model qwen2.5:7b-instruct
"""
from __future__ import annotations

import argparse
import json

import pandas as pd
from tqdm import tqdm

from src import config
from src.answer_eval import load
from src.evaluate import Judge
from src.llm import get_chat
from src.rag import SupportRAG

OUT = config.RESULTS_DIR / "judge_sample.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--model", default="qwen2.5:7b-instruct")
    args = ap.parse_args()

    g = load("grounded")
    sample = g.sample(min(args.n, len(g)), random_state=config.SEED)
    rag = SupportRAG(gate=None)
    judge = Judge(llm=get_chat(args.model))
    done = {json.loads(l)["ticket_id"] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
    with OUT.open("a") as f:
        for _, r in tqdm(sample[~sample["ticket_id"].isin(done)].iterrows(), total=len(sample) - len(done)):
            ctx = [d.page_content for d, _ in rag.retrieve(r["query"], "grounded")]
            try:
                faith, claims = judge.faithfulness(r["answer"], ctx)
            except Exception as e:
                tqdm.write(f"  ! {r['ticket_id']}: {type(e).__name__}")
                continue
            f.write(json.dumps({"ticket_id": r["ticket_id"], "faithfulness": faith, "n_claims": len(claims), "judge_model": args.model}) + "\n")
            f.flush()
    summarise()


def summarise() -> dict | None:
    if not OUT.exists():
        return None
    j = pd.read_json(OUT, lines=True).merge(load("grounded")[["ticket_id", "gate_abstain"]], on="ticket_id")
    j["hall"] = j["faithfulness"] < config.HALLUCINATION_THRESHOLD
    rate = lambda d: round(float(d["hall"].mean()), 4) if len(d) else float("nan")
    res = {"judge_model": j["judge_model"].iloc[0], "n": int(len(j)), "mean_faithfulness": round(float(j["faithfulness"].mean()), 4),
           "hallucination_rate": rate(j), "hallucination_rate_gate_answered": rate(j[~j["gate_abstain"]]),
           "hallucination_rate_gate_escalated": rate(j[j["gate_abstain"]]),
           "n_gate_answered": int((~j["gate_abstain"]).sum()), "n_gate_escalated": int(j["gate_abstain"].sum())}
    (config.RESULTS_DIR / "judge_sample.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
