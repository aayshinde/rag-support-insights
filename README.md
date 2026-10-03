# RAG Support Insights & LLM Evaluation

**[Live interactive demo](https://aayshinde.github.io/rag-support-insights/)** | measured results in [`results/`](results/) | a support-ticket triage system that escalates to a human when its evidence is weak.

A retrieval-augmented pipeline that triages customer-support messages: it retrieves similar
past tickets from a FAISS index, classifies the new issue (category + intent), and recommends a
**next-best-action** grounded in how those past tickets were resolved. A RAGAS-style evaluation
harness measures relevance and faithfulness on held-out queries and compares a naive RAG
baseline against the grounded version that's served by the API and dashboard.

**Stack:** Python · LangChain · OpenAI API · FAISS · FastAPI · Streamlit

```
customer message
      │
      ▼
 OpenAI embeddings ──► FAISS (≈28K past tickets: query + resolution)
                              │ top-k by cosine similarity
                              ▼
               similarity threshold (grounded only)
                              │
                              ▼
     LLM w/ structured output ──► category · intent · next-best-action
     (numbered sources, cite-or-escalate)     steps · cited sources · confidence
                              │
          ┌───────────────────┴───────────────────┐
          ▼                                       ▼
   FastAPI /recommend /classify          Streamlit triage + eval dashboard
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # add your OPENAI_API_KEY

python -m src.data --source both      # download + clean + split (prints record counts)
python -m src.ingest                  # embed knowledge base into FAISS  (--limit 2000 for a cheap dry run)
python -m src.rag "I was charged twice for my order"   # try one query
python -m src.evaluate                # 200 held-out queries, naive vs grounded -> results/report.md

uvicorn api:app --reload              # http://localhost:8000/docs
streamlit run app.py                  # http://localhost:8501
```

Offline tests (no API key needed): `pytest -q` (20 tests)

## Data

| Source | Rows (approx.) | Mapping |
|---|---|---|
| [Bitext customer support](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) | ~26.9K | instruction → query, category, intent, response → resolution |
| [Customer support tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) (English) | varies by version | subject+body → query, queue → category, type → intent, answer → resolution |

`--source both` concatenates them, de-duplicates, and holds out 200 records (stratified by
category) that are **never indexed**, so evaluation queries can't just retrieve themselves.
`python -m src.data` prints the exact counts — use those numbers, not the approximations above.
You can also bring your own CSV (`--source csv --csv file.csv` with `query, category, resolution`
and optionally `intent`).

## Pipeline variants

| | naive (v1) | grounded (v2, deployed) |
|---|---|---|
| Retrieval | top-k | top-k, drop matches below `MIN_SIMILARITY` |
| Prompt | "use the context to help" | answer only from numbered sources, no invented policies/timeframes/URLs |
| Category | free text | constrained to known labels, kNN-vote fallback |
| Unsupported issue | guesses | `escalate=true`, low confidence |
| Output | structured (same schema) | structured + cited source numbers |

## Evaluation methodology

`src/evaluate.py` runs both variants on the same held-out queries and uses an LLM judge
(`JUDGE_MODEL`):

- **Faithfulness** — RAGAS method: the answer is decomposed into atomic claims; each is checked
  against the retrieved context. Score = supported claims / total claims.
- **Hallucination** — a response with faithfulness below `HALLUCINATION_THRESHOLD` (0.7).
- **Relevance** — judge scores 1–5 how well the answer addresses the customer's message;
  *relevance rate* = share scoring ≥ 4.
- **Classification** — category and intent accuracy vs. the held-out gold labels, plus a non-LLM
  kNN-vote baseline.
- **Retrieval hit@k** — whether any retrieved ticket shares the gold intent.

**Hallucination reduction** = (naive rate − grounded rate) / naive rate.

Results go to `results/report.md`, `results/summary.json` and per-query JSONL files (with every
claim's verdict). Runs are resumable — rerun the command after an interruption and it skips
finished queries; `--fresh` starts over. The Streamlit *Evaluation* tab lets you browse
hallucinated answers claim by claim.

### Results

Measured with a local stack (`llama3.2:3b`, `nomic-embed-text`) on the full 48,178-record index.
The LLM-judge comparison (`python -m src.evaluate`) was not run with the small local judge; the
reliability study below uses objective gold labels instead, so it needs no judge.

## Reliability layer: knowing when not to answer

`src/reliability.py` turns retrieval evidence (top-k similarity, neighbour agreement, vote entropy) into a
calibrated probability that the triage is correct, and escalates to a human otherwise. It is a two-stage gate:

1. **Relevance guard.** If the best match's similarity is below a cut-off, nothing in the knowledge base resembles the message, so escalate.
2. **Correctness gate.** Gradient boosting predicts P(triage correct) from agreement features; below the threshold, escalate.

The LLM call is skipped on escalation. The model is fit on 4,000 leave-one-out knowledge-base queries and tested on
200 held-out queries that were never indexed. Correctness is judged against gold labels, so no LLM judge is needed.

| | answer everything | with the gate |
|---|---|---|
| Held-out category accuracy on answered queries | 81.5% | **91.4%** (87% of queries answered, 95% CI for error 4.6% to 12.6%) |
| Free-text tickets only (n=98) | 62.2% correct | 73% answered, 20.8% of those wrong |
| Out-of-scope messages (CLINC) | answers everything | **98% escalated** |

### Experiments and what they showed

| Question | Result | Where |
|---|---|---|
| Is the learned gate better than simpler methods? | Gradient boosting beats logistic regression, vote share, entropy and raw similarity (AURC 0.029 vs 0.038 / 0.041 / 0.044 / 0.255). Split conformal prediction holds its coverage guarantee but is no better than a vote-share threshold. | `results/baselines.md` |
| Does it escalate text the KB should never answer? | The correctness gate alone escalated only 28% of out-of-scope messages. Raw similarity, useless for ranking correctness, separates unrelated text almost perfectly (AUROC 0.99+), so it became the relevance guard in front. | `results/stress_test.md` |
| Does it survive typos? | No. 8% character noise costs about 6 points of accuracy and the error on answered queries rises from 8.7% to 15.5%. Training on noisy copies did not help, so it is not used. Open weakness. | `results/stress_test.md` |
| Does the LLM know when it is wrong? | No. Its self-reported confidence takes 2 distinct values (AUROC 0.55) against 0.88 for the retrieval-based score (n=100). | `results/confidence_study.json` |
| Are the written answers good? | Category accuracy 92.4% on answered vs 69.8% on escalated queries. But copying the top retrieved resolution is closer to the human answer (0.91 similarity) than the LLM rewrite (0.81), so on this data generation adds structure more than accuracy. | `results/answer_eval.md` |
| Where does triage fail? | 57% of all errors are confusions among 4 overlapping categories (PRODUCT SUPPORT, TECHNICAL SUPPORT, IT SUPPORT, CUSTOMER SERVICE). The taxonomy, not the retriever, is the main problem. | `results/failure_analysis.md` |
| Why does raw similarity rank backwards? | Simpson's paradox: easy templated messages are short and score lower similarity than long, hard free-text tickets. | `results/reliability.md` |

Reproduce:
```bash
python -m src.reliability            # fit the gate, write the report and plots
python -m src.baselines              # conformal, boosting, isotonic, single-feature baselines
python -m src.stress_test            # CLINC / Banking77 out-of-domain text and typo noise
python -m src.failure_analysis       # confusion pairs and the taxonomy-merge experiment
python -m src.answer_eval --variants grounded --n 200     # generate and score answers (slow, local LLM)
python -m src.confidence_study --n 100
python -m src.export_demo && python -m src.build_demo     # rebuild the static demo in docs/
pytest -q                            # 20 offline tests, also run in CI
```

**Limits:** 200 held-out queries gives a wide interval; half the data is templated and trivially easy; the main
label is neighbour-vote category correctness and no human has graded answer wording (a blank sheet is generated by
`python -m src.answer_eval --make-human-sheet 50`); the generator is a 3B model; typo robustness is unsolved; and
the Dockerfile and compose file are included but were not built or run on the development machine.

### Deployment notes
`Dockerfile` and `docker-compose.yml` run the API with Ollama (`PROVIDER=ollama`, `OLLAMA_BASE_URL`). GitHub Actions
(`.github/workflows/ci.yml`) runs the offline tests on every push. `python -m src.benchmark` measures latency and the
share of LLM calls the gate avoids, and `python -m src.judge_sample` scores faithfulness with a 7B judge. Neither was run on the
8 GB development laptop: it ran out of memory and the 7B model froze it, so those two results are not reported. A grounded-vs-naive
prompt comparison (`--variants naive`) was also cut short and is omitted. LLM output length is capped (`MAX_NEW_TOKENS`) after a
3B model looped for thousands of tokens on one query.

## Configuration (`.env`)

| Variable | Default | Notes |
|---|---|---|
| `CHAT_MODEL` | `gpt-4o-mini` | set `gpt-3.5-turbo` if you want to match that model exactly |
| `JUDGE_MODEL` | `gpt-4o-mini` | ideally stronger than the chat model |
| `PROVIDER` | `openai` | set `ollama` for a free local setup (`CHAT_MODEL=llama3.2:3b`, `EMBED_MODEL=nomic-embed-text`) |
| `EMBED_MODEL` | `text-embedding-3-small` | `local` = offline hashing embedder for tests |
| `TOP_K` | 5 | retrieved records per query |
| `MIN_SIMILARITY` | 0.35 | cosine cut-off in the grounded variant |
| `HALLUCINATION_THRESHOLD` | 0.7 | |

## Project layout

```
src/config.py     settings
src/data.py       download, normalise, de-dup, stratified hold-out split
src/llm.py        model factories (+ offline embedder)
src/ingest.py     FAISS index build / load
src/rag.py        retrieval, prompts, structured output, naive vs grounded
src/reliability.py two-stage gate, risk-coverage analysis, knowledge gaps
src/baselines.py, stress_test.py, failure_analysis.py, answer_eval.py, confidence_study.py, judge_sample.py, benchmark.py   experiments
src/evaluate.py   LLM-judge faithfulness/relevance, classification metrics, report
api.py            FastAPI: /health /recommend /classify /metrics
app.py            Streamlit triage UI + evaluation explorer
tests/            offline end-to-end tests
```

## Limitations

- LLM-as-judge scores are themselves model outputs; spot-check a sample by hand (the dashboard
  shows every claim verdict) and report the judge model alongside the numbers.
- Bitext responses are templated, so retrieval is easier than on messy real tickets; the
  tickets dataset adds noisier examples.
- 200 queries gives roughly ±5–6 percentage points of sampling error on a rate near 80%.
