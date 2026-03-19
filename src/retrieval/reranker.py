"""Cohere reranking for the retrieval pipeline."""
import asyncio
import logging
import os

import cohere

from src.tracing import get_tracer, set_llm_attributes, timed_span

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
    tracer = get_tracer()
    with timed_span(tracer, "retrieval.rerank", {
        "rerank.input_count": len(chunks),
        "rerank.top_n": top_n,
    }) as span:
        if not chunks:
            span.set_attribute("rerank.output_count", 0)
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
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            span.set_attribute("rerank.fallback", True)
            fallback = [{**c, "rerank_score": 0.0} for c in chunks[:top_n]]
            span.set_attribute("rerank.output_count", len(fallback))
            return fallback

        set_llm_attributes(span, model=RERANK_MODEL)
        span.set_attribute("rerank.fallback", False)

        # Map reranked results back to chunk dicts; response.results has .index and .relevance_score
        out: list[dict] = []
        for r in response.results:
            chunk = chunks[r.index]
            out.append({**chunk, "rerank_score": float(r.relevance_score)})
        result = out[:top_n]
        span.set_attribute("rerank.output_count", len(result))
        if result:
            span.set_attribute("rerank.top_relevance_score", result[0]["rerank_score"])
            scores = [r["rerank_score"] for r in result]
            span.set_attribute("rerank.mean_relevance_score", sum(scores) / len(scores))
        return result
