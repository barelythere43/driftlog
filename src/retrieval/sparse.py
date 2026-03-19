"""BM25 sparse search over chunk content."""
import logging
import re

from rank_bm25 import BM25Okapi
from sqlalchemy import select

from src.database import async_session_factory
from src.models import Chunk
from src.tracing import get_tracer, timed_span

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on whitespace, strip punctuation. No stemming."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Index:
    """In-memory BM25 index over chunks. Build from DB, then search."""

    def __init__(self) -> None:
        self._built = False
        self._chunks: list[dict] = []
        self._corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    @property
    def is_built(self) -> bool:
        return self._built

    async def build_index(self) -> None:
        """Load all chunks from PostgreSQL, tokenize content, build BM25 index in memory."""
        async with async_session_factory() as session:
            result = await session.execute(select(Chunk))
            rows = result.scalars().all()
            self._chunks = [
                {
                    "chunk_id": str(row.id),
                    "content": row.content,
                    "document_id": str(row.document_id),
                    "metadata": dict(row.metadata_) if row.metadata_ else {},
                }
                for row in rows
            ]
            self._corpus = [_tokenize(row.content) for row in rows]
        if self._corpus:
            self._bm25 = BM25Okapi(self._corpus)
        else:
            self._bm25 = None
        self._built = True

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """Return top_k chunks by BM25 score. Same shape as dense_search (score instead of similarity_score)."""
        tracer = get_tracer()
        with timed_span(tracer, "retrieval.sparse_search", {"search.top_k": top_k}) as span:
            if not self._built or self._bm25 is None:
                logger.warning("BM25 index not built — call build_index() first")
                span.set_attribute("search.results_count", 0)
                span.set_attribute("search.index_built", False)
                return []
            span.set_attribute("search.index_built", True)
            span.set_attribute("search.corpus_size", len(self._chunks))
            query_tokens = _tokenize(query)
            if not query_tokens:
                span.set_attribute("search.results_count", 0)
                return []
            span.set_attribute("search.query_token_count", len(query_tokens))
            scores = self._bm25.get_scores(query_tokens)
            indexed = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            results = [
                {
                    "chunk_id": self._chunks[i]["chunk_id"],
                    "content": self._chunks[i]["content"],
                    # BM25 score — fusion layer normalizes this with dense's similarity_score
                    "score": float(scores[i]),
                    "document_id": self._chunks[i]["document_id"],
                    "metadata": self._chunks[i]["metadata"],
                }
                for i in indexed
            ]
            span.set_attribute("search.results_count", len(results))
            if results:
                span.set_attribute("search.top_bm25_score", results[0]["score"])
            return results


bm25_index = BM25Index()
