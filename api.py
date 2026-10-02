"""FastAPI service exposing the grounded RAG pipeline.

Run:  uvicorn api:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import json
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import config

app = FastAPI(title="RAG Support Insights", version="1.0.0")


class Query(BaseModel):
    message: str = Field(min_length=3, examples=["I was charged twice for order 55123"])
    variant: Literal["grounded", "naive"] = "grounded"


class ClassifyResponse(BaseModel):
    category: str
    intent: str
    knn_category: str | None
    confidence: float
    calibrated_confidence: float | None = None
    abstained: bool = False


def _rag():
    from src.rag import get_rag

    if not (config.INDEX_DIR / "index.faiss").exists():
        raise HTTPException(503, "Index not built. Run `python -m src.ingest` first.")
    return get_rag()


@app.get("/health")
def health():
    meta_path = config.INDEX_DIR / "index_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    return {"status": "ok", "index_ready": meta is not None, "index": meta, "chat_model": config.CHAT_MODEL}


@app.post("/recommend")
def recommend(q: Query):
    """Full pipeline: category + intent + next-best-action + supporting records."""
    return _rag().run(q.message, variant=q.variant).to_dict()


@app.post("/classify", response_model=ClassifyResponse)
def classify(q: Query):
    r = _rag().run(q.message, variant=q.variant)
    return ClassifyResponse(
        category=r.insight["category"],
        intent=r.insight["intent"],
        knn_category=r.knn_category,
        confidence=r.insight["confidence"],
        calibrated_confidence=r.calibrated_confidence,
        abstained=r.abstained,
    )


@app.get("/reliability")
def reliability():
    """Confidence-gate evaluation: risk-coverage trade-off and knowledge-gap clusters."""
    p = config.RESULTS_DIR / "reliability.json"
    if not p.exists():
        raise HTTPException(404, "Run `python -m src.reliability` first.")
    return json.loads(p.read_text())


@app.get("/metrics")
def metrics():
    """Latest offline evaluation summary."""
    p = config.RESULTS_DIR / "summary.json"
    if not p.exists():
        raise HTTPException(404, "No evaluation yet. Run `python -m src.evaluate`.")
    return json.loads(p.read_text())
