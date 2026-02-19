from openai import AsyncOpenAI

EMBEDDING_MODEL = "text-embedding-3-small"


async def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Embed a list of text chunks in one batched API call. Returns embeddings in same order as input."""
    if not chunks:
        return []
    client = AsyncOpenAI()
    response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=chunks,
    )
    return [item.embedding for item in response.data]


async def embed_text(text: str) -> list[float]:
    """Embed a single string (e.g. a search query). Same model and API as embed_chunks."""
    embeddings = await embed_chunks([text])
    return embeddings[0] if embeddings else []
