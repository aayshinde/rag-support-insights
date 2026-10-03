"""Central configuration. Everything is overridable through environment variables / .env."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "index"
RESULTS_DIR = ROOT / "results"

RAW_CSV = DATA_DIR / "support_tickets.csv"   # full dataset
KB_CSV = DATA_DIR / "kb_train.csv"           # records embedded into FAISS
TEST_CSV = DATA_DIR / "eval_holdout.csv"     # held-out records never indexed

# "openai" (default) or "ollama" for a free local setup
PROVIDER = os.getenv("PROVIDER", "openai")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")  # e.g. http://ollama:11434 inside docker-compose; default is localhost

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

TOP_K = int(os.getenv("TOP_K", "5"))
# Retrieved chunks with a cosine similarity below this are dropped in the
# "grounded" pipeline so the model is not fed irrelevant context.
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.35"))

N_EVAL_QUERIES = int(os.getenv("N_EVAL_QUERIES", "200"))
# A response whose faithfulness score falls below this counts as a hallucination.
HALLUCINATION_THRESHOLD = float(os.getenv("HALLUCINATION_THRESHOLD", "0.7"))
SEED = int(os.getenv("SEED", "42"))

for d in (DATA_DIR, INDEX_DIR, RESULTS_DIR):
    d.mkdir(exist_ok=True)
