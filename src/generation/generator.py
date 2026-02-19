import re

from anthropic import AsyncAnthropic

from src.config import settings
from src.generation.prompts import build_prompt

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 1024
CITATION_PATTERN = re.compile(r"\[(\d+)\]")
EXCERPT_MAX_LEN = 150

_client: AsyncAnthropic | None = None


def _get_client() -> AsyncAnthropic:
    """Return a cached AsyncAnthropic client, creating it on first use."""
    global _client
    if _client is None:
        _client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _client


async def generate_answer(question: str, chunks: list[dict]) -> dict:
    """Call Claude with context from chunks, parse response, and return answer with citations and confidence."""
    prompt = build_prompt(question, chunks)
    client = _get_client()
    response = await client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=0,
        system=prompt["system"],
        messages=[{"role": "user", "content": prompt["user"]}],
    )
    answer = response.content[0].text if response.content else ""

    cited_indices = sorted(
        set(int(m) for m in CITATION_PATTERN.findall(answer) if m.isdigit())
    )
    cited_indices = [i for i in cited_indices if 1 <= i <= len(chunks)]

    citations = []
    cited_scores: list[float] = []
    for i in cited_indices:
        chunk = chunks[i - 1]
        meta = chunk.get("metadata") or {}
        content = chunk.get("content", "")
        excerpt = content[:EXCERPT_MAX_LEN] + ("..." if len(content) > EXCERPT_MAX_LEN else "")
        citations.append(
            {
                "chunk_id": chunk.get("chunk_id", ""),
                "source": meta.get("source") or "Unknown",
                "location": meta.get("location") or "N/A",
                "excerpt": excerpt,
            }
        )
        if "rerank_score" in chunk:
            cited_scores.append(chunk["rerank_score"])

    if cited_scores:
        confidence = sum(cited_scores) / len(cited_scores)
    else:
        confidence = 0.0

    return {
        "answer": answer,
        "confidence": round(confidence, 4),
        "citations": citations,
        "query_type": "factual",
        "chunks_retrieved": len(chunks),
        "chunks_after_rerank": len(chunks),
    }
