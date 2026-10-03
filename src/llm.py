"""Model factories. Setting EMBED_MODEL=local swaps OpenAI embeddings for a
deterministic hashing embedder so the index can be built and tested offline."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from langchain_core.embeddings import Embeddings

from src import config


class LocalHashEmbeddings(Embeddings):
    """TF-hashing embedder (no API calls). Good enough for smoke tests; use OpenAI for real runs."""

    def __init__(self, n_features: int = 1024):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.vec = HashingVectorizer(n_features=n_features, ngram_range=(1, 2), alternate_sign=False, norm="l2")

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self.vec.transform(texts).toarray().astype(np.float32).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


class NormalizedEmbeddings(Embeddings):
    """Wraps an embedder and L2-normalises vectors so inner product == cosine similarity."""

    def __init__(self, inner: Embeddings):
        self.inner = inner

    @staticmethod
    def _norm(vecs: list[list[float]]) -> list[list[float]]:
        a = np.asarray(vecs, dtype=np.float32)
        a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
        return a.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 32):
            out += self.inner.embed_documents([t[:2000] for t in texts[i : i + 32]])
        return self._norm(out)

    def embed_query(self, text: str) -> list[float]:
        return self._norm([self.inner.embed_query(text[:2000])])[0]


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    if config.EMBED_MODEL == "local":
        return LocalHashEmbeddings()
    if config.PROVIDER == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return NormalizedEmbeddings(OllamaEmbeddings(model=config.EMBED_MODEL, base_url=config.OLLAMA_BASE_URL))
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=config.EMBED_MODEL, chunk_size=500)


@lru_cache(maxsize=4)
def get_chat(model: str | None = None, temperature: float = 0.0):
    if config.PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model or config.CHAT_MODEL, temperature=temperature, base_url=config.OLLAMA_BASE_URL)
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model or config.CHAT_MODEL, temperature=temperature, max_retries=5, timeout=60)
