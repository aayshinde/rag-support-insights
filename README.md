# RAG Support Insights & LLM Evaluation

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

Offline tests (no API key needed): `pytest -q`

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

> Fill this in from `results/report.md` after your run.

| Metric | naive | grounded |
|---|---|---|
| Relevance rate | | |
| Hallucination rate | | |
| Category accuracy | | |

## Configuration (`.env`)

| Variable | Default | Notes |
|---|---|---|
| `CHAT_MODEL` | `gpt-4o-mini` | set `gpt-3.5-turbo` if you want to match that model exactly |
| `JUDGE_MODEL` | `gpt-4o-mini` | ideally stronger than the chat model |
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
