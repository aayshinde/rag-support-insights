"""RAG pipeline: retrieve similar past support records, then classify the new
issue and recommend a next-best-action grounded in how those records were resolved.

Two variants are kept side by side so their hallucination rates can be compared:

  naive     top-k retrieval -> loose prompt.           (v1 baseline)
  grounded  top-k retrieval -> similarity threshold ->  (v2, what the API/UI serve)
            strict "answer only from sources" prompt with numbered citations,
            category constrained to known labels, explicit escalate path when the
            sources don't cover the issue.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Literal

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src import config
from src.ingest import index_categories, load_index
from src.llm import get_chat
from src.reliability import ConfidenceModel

Variant = Literal["naive", "grounded"]


class SupportInsight(BaseModel):
    category: str = Field(description="Issue category; must be one of the allowed categories")
    intent: str = Field(description="Specific intent label, e.g. cancel_order, payment_issue")
    next_best_action: str = Field(description="One-sentence recommended action for the agent")
    steps: list[str] = Field(description="2-5 concrete steps the agent should take or tell the customer")
    cited_sources: list[int] = Field(default_factory=list, description="Source numbers the answer relies on")
    confidence: float = Field(ge=0, le=1, description="0-1 confidence that the sources cover this issue")
    escalate: bool = Field(default=False, description="True if sources don't cover the issue and a human should take it")


@dataclass
class RagResult:
    query: str
    variant: str
    insight: dict
    answer: str                       # flattened text used for evaluation
    contexts: list[str]               # exactly what the model saw
    retrieved: list[dict] = field(default_factory=list)
    knn_category: str | None = None   # majority vote over neighbours (non-LLM baseline)
    calibrated_confidence: float | None = None  # P(triage correct) from retrieval evidence (src.reliability)
    abstained: bool = False           # True when the confidence gate overrode the answer with an escalation

    def to_dict(self) -> dict:
        return asdict(self)


NAIVE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful customer support assistant. Use the context to help answer."),
        (
            "human",
            "Context:\n{context}\n\nCustomer message: {query}\n\n"
            "Classify the issue and recommend what the support agent should do next.",
        ),
    ]
)

GROUNDED_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a support-operations assistant. You receive a new customer message and "
            "numbered SOURCES, each a past support record with how it was resolved.\n"
            "Rules:\n"
            "1. Base the next-best-action and every step ONLY on what the sources state. Do not "
            "invent policies, timeframes, amounts, URLs, phone numbers or product features.\n"
            "2. Cite the source numbers you used in cited_sources.\n"
            "3. category MUST be one of: {categories}.\n"
            "4. If the sources do not cover this issue, set escalate=true, confidence below 0.4, "
            "and recommend routing to a human agent instead of guessing.\n"
            "5. Keep placeholders generic (e.g. 'the order number') rather than copying another "
            "customer's specific details.",
        ),
        ("human", "SOURCES:\n{context}\n\nNEW CUSTOMER MESSAGE:\n{query}"),
    ]
)


def format_sources(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{i}] (category={d.metadata['category']}, intent={d.metadata['intent']})\n{d.page_content}"
        for i, d in enumerate(docs, 1)
    )


def knn_vote(scored: list[tuple[Document, float]]) -> str | None:
    """Similarity-weighted majority vote over neighbour categories."""
    if not scored:
        return None
    weights: dict[str, float] = defaultdict(float)
    for doc, score in scored:
        weights[doc.metadata["category"]] += max(score, 0.0)
    return max(weights, key=weights.get) if any(weights.values()) else Counter(
        d.metadata["category"] for d, _ in scored
    ).most_common(1)[0][0]


def insight_to_text(ins: SupportInsight) -> str:
    lines = [f"Category: {ins.category} / {ins.intent}", f"Next best action: {ins.next_best_action}"]
    lines += [f"- {s}" for s in ins.steps]
    if ins.escalate:
        lines.append("Escalate to a human agent.")
    return "\n".join(lines)


class SupportRAG:
    def __init__(self, store=None, llm=None, categories: list[str] | None = None, top_k: int = config.TOP_K,
                 gate: ConfidenceModel | None | Literal["auto"] = "auto"):
        self.store = store if store is not None else load_index()
        self.llm = llm if llm is not None else get_chat()
        self.categories = categories if categories is not None else index_categories()
        self.top_k = top_k
        # "auto" loads the fitted gate if `python -m src.reliability` has been run; None disables abstention
        self.gate = ConfidenceModel.load() if gate == "auto" else gate
        structured = self.llm.with_structured_output(SupportInsight)
        self.chains = {"naive": NAIVE_PROMPT | structured, "grounded": GROUNDED_PROMPT | structured}

    def retrieve(self, query: str, variant: Variant, scored=None) -> list[tuple[Document, float]]:
        scored = scored if scored is not None else self.store.similarity_search_with_score(query, k=self.top_k)
        if variant == "grounded":
            kept = [(d, s) for d, s in scored if s >= config.MIN_SIMILARITY]
            scored = kept or scored[:1]  # always keep the best match so the model can judge coverage
        return scored

    def run(self, query: str, variant: Variant = "grounded") -> RagResult:
        raw = self.store.similarity_search_with_score(query, k=self.top_k)
        scored = self.retrieve(query, variant, scored=raw)
        docs = [d for d, _ in scored]
        context = format_sources(docs)
        knn = knn_vote(scored)

        confidence, abstained = None, False
        if self.gate is not None and raw:
            _, confidence = self.gate.confidence([s for _, s in raw], [d.metadata["category"] for d, _ in raw])
            abstained = variant == "grounded" and confidence < self.gate.tau

        if abstained:  # evidence too weak: skip the LLM call entirely and hand off to a human
            insight = SupportInsight(
                category=knn or "UNKNOWN",
                intent="unknown",
                next_best_action="Route to a human agent: the knowledge base does not reliably cover this issue.",
                steps=["Escalate to a human agent", "Add the resolved ticket to the knowledge base"],
                confidence=round(confidence, 2),
                escalate=True,
            )
        else:
            inputs = {"context": context, "query": query, "categories": ", ".join(self.categories) or "any"}
            insight = self.chains[variant].invoke(inputs)
            if variant == "grounded" and self.categories and insight.category not in self.categories:
                insight.category = knn or insight.category  # fall back to neighbours if the LLM drifts

        return RagResult(
            query=query,
            variant=variant,
            insight=insight.model_dump(),
            answer=insight_to_text(insight),
            contexts=[d.page_content for d in docs],
            retrieved=[
                {
                    "rank": i,
                    "score": round(float(s), 4),
                    "ticket_id": d.metadata["ticket_id"],
                    "category": d.metadata["category"],
                    "intent": d.metadata["intent"],
                    "query": d.metadata["query"],
                    "resolution": d.metadata["resolution"],
                }
                for i, (d, s) in enumerate(scored, 1)
            ],
            knn_category=knn,
            calibrated_confidence=None if confidence is None else round(confidence, 4),
            abstained=abstained,
        )


@lru_cache(maxsize=1)
def get_rag() -> SupportRAG:
    return SupportRAG()


if __name__ == "__main__":
    import json
    import sys

    q = " ".join(sys.argv[1:]) or "I was charged twice for my order and need my money back"
    print(json.dumps(get_rag().run(q).to_dict(), indent=2)[:4000])
