"""Streamlit dashboard: live triage + evaluation explorer.

Run:  streamlit run app.py
Set API_URL=http://localhost:8000 to call the FastAPI service instead of the in-process pipeline.
"""
from __future__ import annotations

import json
import os

import pandas as pd
import requests
import streamlit as st

from src import config

st.set_page_config(page_title="RAG Support Insights", page_icon="🎧", layout="wide")
API_URL = os.getenv("API_URL")


@st.cache_resource(show_spinner="Loading FAISS index…")
def local_rag():
    from src.rag import SupportRAG

    return SupportRAG()


def run_query(message: str, variant: str) -> dict:
    if API_URL:
        r = requests.post(f"{API_URL}/recommend", json={"message": message, "variant": variant}, timeout=120)
        r.raise_for_status()
        return r.json()
    return local_rag().run(message, variant=variant).to_dict()


st.title("RAG Support Insights")
meta_path = config.INDEX_DIR / "index_meta.json"
if meta_path.exists():
    meta = json.loads(meta_path.read_text())
    st.caption(
        f"{meta['n_documents']:,} support records indexed · {len(meta['categories'])} categories · "
        f"embeddings `{meta['embed_model']}` · chat `{config.CHAT_MODEL}`"
    )

tab_triage, tab_rel, tab_extra, tab_eval = st.tabs(["Triage a ticket", "Reliability & gaps", "Stress & failures", "Evaluation"])

# ------------------------------------------------------------------ triage
with tab_triage:
    examples = [
        "I was charged twice for my last order and want one of them refunded",
        "how do I change the shipping address, the package hasn't left yet",
        "your app keeps logging me out every time I open it",
        "I want to delete my account and all my data",
    ]
    c1, c2 = st.columns([3, 1])
    with c2:
        variant = st.radio("Pipeline", ["grounded", "naive"], help="grounded = deployed version")
        ex = st.selectbox("Example", ["—"] + examples)
    with c1:
        msg = st.text_area("Customer message", value="" if ex == "—" else ex, height=120)
        go = st.button("Analyse", type="primary", disabled=len(msg.strip()) < 3)

    if go:
        if not meta_path.exists() and not API_URL:
            st.error("Index not built yet — run `python -m src.ingest`.")
            st.stop()
        with st.spinner("Retrieving similar tickets and drafting a recommendation…"):
            res = run_query(msg, variant)
        ins = res["insight"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Category", ins["category"])
        m2.metric("Intent", ins["intent"])
        cc = res.get("calibrated_confidence")
        m3.metric("Calibrated P(correct)" if cc is not None else "LLM confidence",
                  f"{(cc if cc is not None else ins['confidence']):.0%}")
        m4.metric("kNN vote", res.get("knn_category") or "—")
        if res.get("abstained"):
            why = ("nothing in the knowledge base resembles this message (relevance guard)"
                   if res.get("abstain_reason") == "out_of_domain"
                   else "the retrieved tickets disagree on how to handle it (confidence gate)")
            st.error(f"Escalated to a human without calling the LLM: {why}.")
        elif ins["escalate"]:
            st.warning("Sources don't cover this issue well — route to a human agent.")
        st.subheader("Next best action")
        st.write(ins["next_best_action"])
        for s in ins["steps"]:
            st.markdown(f"- {s}")
        if ins["cited_sources"]:
            st.caption("Cites sources " + ", ".join(f"[{i}]" for i in ins["cited_sources"]))
        st.subheader("Retrieved support records")
        for r in res["retrieved"]:
            with st.expander(f"[{r['rank']}] {r['category']} / {r['intent']} · similarity {r['score']:.2f}"):
                st.markdown(f"**Customer:** {r['query']}")
                st.markdown(f"**Resolution:** {r['resolution']}")

# ------------------------------------------------------------------ reliability
with tab_rel:
    rp = config.RESULTS_DIR / "reliability.json"
    if not rp.exists():
        st.info("Run `python -m src.reliability` to fit the confidence gate and generate this view.")
    else:
        rel = json.loads(rp.read_text())
        h = rel["at_threshold"]["holdout"]
        base = rel["baseline_accuracy"]["holdout"]
        st.markdown(
            f"**When should the system abstain?** Answering everything is wrong {1 - base:.0%} of the time. "
            f"The gate answers **{h['coverage']:.0%}** of held-out queries at **{h['risk']:.1%}** error "
            f"(target ≤ {rel['target_risk']:.0%}) and escalates the rest."
        )
        r1, r2, r3 = st.columns(3)
        r1.metric("Coverage (answered)", f"{h['coverage']:.0%}")
        r2.metric("Error on answered", f"{h['risk']:.1%}", f"{(h['risk'] - (1 - base)) * 100:+.1f} pts vs answer-all",
                  delta_color="inverse")
        r3.metric("Abstention threshold", f"{rel['tau']:.2f}")
        if (config.RESULTS_DIR / "reliability.png").exists():
            st.image(str(config.RESULTS_DIR / "reliability.png"))
        st.subheader("Per-source behaviour (held-out)")
        st.dataframe(pd.DataFrame(rel["by_source"]["holdout"]).T, use_container_width=True)
        st.subheader("Knowledge gaps")
        st.caption("Clusters of queries the gate abstains on. Low category purity means the same topic is filed "
                   "under many categories — a taxonomy problem, not just missing knowledge.")
        for i, gp in enumerate(rel["knowledge_gaps"], 1):
            with st.expander(f"{i}. {', '.join(gp['keywords'][:3])} — {gp['size']} queries, "
                             f"purity {gp['category_purity']:.0%} {gp['top_gold_category']}"):
                for e in gp["examples"]:
                    st.markdown(f"> {e}")

# ------------------------------------------------------------------ stress tests, baselines, failures
with tab_extra:
    def show_md(name: str, empty: str):
        p = config.RESULTS_DIR / name
        st.markdown(p.read_text().replace("(reliability.png)", "").replace("![baselines](baselines.png)", "")) if p.exists() else st.info(empty)

    sub1, sub2, sub3, sub4 = st.tabs(["Stress tests", "Baselines", "Failure analysis", "Answer quality"])
    with sub1:
        show_md("stress_test.md", "Run `python -m src.stress_test`.")
    with sub2:
        show_md("baselines.md", "Run `python -m src.baselines`.")
        if (config.RESULTS_DIR / "baselines.png").exists():
            st.image(str(config.RESULTS_DIR / "baselines.png"))
    with sub3:
        show_md("failure_analysis.md", "Run `python -m src.failure_analysis`.")
    with sub4:
        show_md("answer_eval.md", "Run `python -m src.answer_eval --variants grounded`.")

# ------------------------------------------------------------------ evaluation
with tab_eval:
    sp = config.RESULTS_DIR / "summary.json"
    if not sp.exists():
        st.info("No evaluation results yet. Run `python -m src.evaluate`.")
    else:
        summary = json.loads(sp.read_text())
        variants = summary["variants"]
        g = variants.get("grounded") or next(iter(variants.values()))
        n = variants.get("naive")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Relevance rate", f"{g['relevance_rate']:.0%}",
                  f"{(g['relevance_rate'] - n['relevance_rate']) * 100:+.1f} pts vs naive" if n else None)
        k2.metric("Hallucination rate", f"{g['hallucination_rate']:.1%}",
                  f"{(g['hallucination_rate'] - n['hallucination_rate']) * 100:+.1f} pts vs naive" if n else None,
                  delta_color="inverse")
        k3.metric("Mean faithfulness", f"{g['mean_faithfulness']:.2f}")
        k4.metric("Category accuracy", f"{g['category_acc']:.0%}")
        if "hallucination_reduction" in summary:
            st.success(f"Relative hallucination reduction, naive → grounded: **{summary['hallucination_reduction']:.0%}** "
                       f"on {summary['n_queries']} held-out queries")

        st.dataframe(pd.DataFrame(variants).T, use_container_width=True)

        frames = []
        for v in variants:
            p = config.RESULTS_DIR / f"eval_{v}.jsonl"
            if p.exists():
                frames.append(pd.read_json(p, lines=True))
        if frames:
            df = pd.concat(frames, ignore_index=True)
            st.subheader("Faithfulness distribution")
            bins = pd.cut(df["faithfulness"], [-0.01, 0.2, 0.4, 0.6, 0.8, 1.0])
            dist = df.groupby([bins, "variant"], observed=False).size().unstack(fill_value=0)
            dist.index = dist.index.astype(str)
            st.bar_chart(dist)

            st.subheader("Accuracy by gold category (grounded)")
            gdf = df[df["variant"] == "grounded"] if "grounded" in variants else df
            st.bar_chart(gdf.groupby("gold_category")[["category_correct", "relevant"]].mean())

            st.subheader("Inspect individual queries")
            only_bad = st.checkbox("Only hallucinations")
            view = df[df["hallucination"]] if only_bad else df
            st.dataframe(
                view[["variant", "query", "gold_category", "pred_category", "relevance_score", "faithfulness"]],
                use_container_width=True, height=300,
            )
            if len(view):
                pick = st.selectbox("Row", view.index, format_func=lambda i: f"{view.at[i, 'variant']} · {view.at[i, 'query'][:80]}")
                row = view.loc[pick]
                st.markdown(f"**Answer**\n\n```\n{row['answer']}\n```")
                st.markdown("**Claim check**")
                for c in row["claims"]:
                    st.markdown(f"{'✅' if c['supported'] else '❌'} {c['claim']}")
