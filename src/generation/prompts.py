SYSTEM_PROMPT = """You are a travel knowledge assistant that answers questions using ONLY the provided context below. Do not use any external knowledge.

Rules:
- Base your answer strictly on the context. Do not add information from outside the context.
- Cite every factual claim with the bracket notation used in the context (e.g. [1], [2]). Use the same number that appears next to the passage you are citing.
- If the context does not contain enough information to answer the question, say explicitly: "I don't have enough information to answer that."
- Never fabricate or guess information. If you are unsure, say so."""

USER_PROMPT_TEMPLATE = """{context}

Question: {question}"""


def build_prompt(question: str, chunks: list[dict]) -> dict[str, str]:
    """Format chunks with numeric citations and return system + user prompt dict for response generation."""
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata") or {}
        source = meta.get("source") or "Unknown"
        location = meta.get("location") or "N/A"
        content = chunk.get("content", "")
        block = f"[{i}]\n{content}\nSource: {source} | Location: {location}"
        context_parts.append(block)
    context = "\n\n".join(context_parts) if context_parts else "(No context provided.)"
    return {
        "system": SYSTEM_PROMPT,
        "user": USER_PROMPT_TEMPLATE.format(context=context, question=question),
    }
