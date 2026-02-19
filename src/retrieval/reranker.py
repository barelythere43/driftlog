"""Cohere reranking for the retrieval pipeline."""
import asyncio
import logging
import os

import cohere

logger = logging.getLogger(__name__)

_client: cohere.ClientV2 | None = None

RERANK_MODEL = "rerank-v3.5"


def _get_client() -> cohere.ClientV2:
    global _client
    if _client is None:
        _client = cohere.ClientV2(api_key=os.environ.get("CO_API_KEY"))
    return _client


async def rerank(
    query: str,
    chunks: list[dict],
    top_n: int = 5,
) -> list[dict]:
    """Rerank fused chunks with Cohere, return top_n with rerank_score. Falls back to first top_n on API error."""
    if not chunks:
        return []
    documents = [c["content"] for c in chunks]
    try:
        client = _get_client()
        response = await asyncio.to_thread(
            client.rerank,
            model=RERANK_MODEL,
            query=query,
            documents=documents,
            top_n=min(top_n, len(chunks)),
        )
    except Exception as e:
        logger.exception("Cohere rerank failed, falling back to RRF order: %s", e)
        return [{**c, "rerank_score": 0.0} for c in chunks[:top_n]]

    # Map reranked results back to chunk dicts; response.results has .index and .relevance_score
    out: list[dict] = []
    for r in response.results:
        chunk = chunks[r.index]
        out.append({**chunk, "rerank_score": float(r.relevance_score)})
    return out[:top_n]
