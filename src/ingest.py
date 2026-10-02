"""Embed the knowledge-base split into a FAISS index.

Each support record becomes one Document:
    page_content = customer query + agent resolution  (what gets embedded)
    metadata     = ticket_id, category, intent, resolution, query

Usage:  python -m src.ingest [--limit 5000]
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from tqdm import tqdm

from src import config
from src.llm import get_embeddings

BATCH = 1000


def row_to_doc(row: pd.Series) -> Document:
    text = f"Customer issue: {row['query']}\nResolution: {row['resolution']}"
    return Document(
        page_content=text,
        metadata={
            "ticket_id": row["ticket_id"],
            "category": row["category"],
            "intent": row["intent"],
            "query": row["query"],
            "resolution": row["resolution"],
        },
    )


def build_index(df: pd.DataFrame) -> FAISS:
    docs = [row_to_doc(r) for _, r in df.iterrows()]
    emb = get_embeddings()
    store: FAISS | None = None
    for i in tqdm(range(0, len(docs), BATCH), desc="Embedding", unit="batch"):
        chunk = docs[i : i + BATCH]
        if store is None:
            # OpenAI embeddings (and LocalHashEmbeddings) are unit-length, so inner
            # product == cosine similarity and scores are directly interpretable.
            store = FAISS.from_documents(chunk, emb, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
        else:
            store.add_documents(chunk)
    assert store is not None, "no documents to index"
    return store


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="index only the first N records (cheap dry run)")
    args = ap.parse_args()

    df = pd.read_csv(config.KB_CSV)
    if args.limit:
        df = df.head(args.limit)

    t0 = time.time()
    store = build_index(df)
    store.save_local(str(config.INDEX_DIR))
    meta = {
        "n_documents": len(df),
        "embed_model": config.EMBED_MODEL,
        "categories": sorted(df["category"].unique().tolist()),
        "build_seconds": round(time.time() - t0, 1),
    }
    (config.INDEX_DIR / "index_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Indexed {len(df):,} records into {config.INDEX_DIR} in {meta['build_seconds']}s")


def load_index() -> FAISS:
    return FAISS.load_local(
        str(config.INDEX_DIR),
        get_embeddings(),
        allow_dangerous_deserialization=True,  # our own file, written by build_index
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )


def index_categories() -> list[str]:
    meta = config.INDEX_DIR / "index_meta.json"
    return json.loads(meta.read_text())["categories"] if meta.exists() else []


if __name__ == "__main__":
    main()
