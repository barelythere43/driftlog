from openai import AsyncOpenAI

from src.tracing import get_tracer, set_llm_attributes, timed_span

EMBEDDING_MODEL = "text-embedding-3-small"


async def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Embed a list of text chunks in one batched API call. Returns embeddings in same order as input."""
    if not chunks:
        return []
    tracer = get_tracer()
    with timed_span(tracer, "embedding.batch", {"embedding.chunk_count": len(chunks)}) as span:
        client = AsyncOpenAI()
        response = await client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=chunks,
        )
        # OpenAI embeddings response includes usage
        usage = getattr(response, "usage", None)
        total_tokens = getattr(usage, "total_tokens", None) if usage else None
        set_llm_attributes(
            span,
            model=EMBEDDING_MODEL,
            input_tokens=total_tokens,
            total_tokens=total_tokens,
        )
        span.set_attribute("embedding.dimensions", len(response.data[0].embedding) if response.data else 0)
        return [item.embedding for item in response.data]


async def embed_text(text: str) -> list[float]:
    """Embed a single string (e.g. a search query). Same model and API as embed_chunks."""
    embeddings = await embed_chunks([text])
    return embeddings[0] if embeddings else []
