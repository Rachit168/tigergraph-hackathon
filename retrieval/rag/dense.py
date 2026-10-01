"""Production dense-RAG stub. Always empty; unused by the BM25 RAG path."""

from __future__ import annotations

from retrieval.rag.models import RetrievalResult


class DenseRetriever:
    """Placeholder dense retriever. Production RAG does not use this class.

    `TextRetriever` defaults to BM25 (`method="sparse"`). This stub always
    returns no hits so a caller cannot accidentally treat dense RAG as live.
    Optional FastEmbed / numpy dependencies exist only for the separate
    TigerGraph vector experiment path, not for this class.
    """

    def __init__(self, reason: str | None = None) -> None:
        self.available = False
        self.model_name = None
        self.reason = reason or (
            "DenseRetriever is a production stub and is unused by BM25 RAG. "
            "TigerGraph vector search is a separate experiment path."
        )

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        return RetrievalResult(
            query=query,
            method="dense",
            hits=[],
            params={"available": False, "top_k": top_k, "reason": self.reason},
        )
