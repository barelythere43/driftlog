from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.ingestion.embedder import embed_text
from src.tracing import get_tracer, timed_span


async def dense_search(
    session: AsyncSession,
    query: str,
    *,
    top_k: int = 20,
    location: str | None = None,
    country: str | None = None,
    tags: list[str] | None = None,
) -> list[dict]:
    """Run pgvector cosine similarity search on chunks with optional metadata filters."""
    tracer = get_tracer()
    with timed_span(tracer, "retrieval.dense_search", {
        "search.top_k": top_k,
        "search.has_location_filter": location is not None,
        "search.has_country_filter": country is not None,
        "search.has_tags_filter": tags is not None and len(tags) > 0,
    }) as span:
        query_embedding = await embed_text(query)
        if not query_embedding:
            span.set_attribute("search.results_count", 0)
            return []
        vector_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        conditions = []
        params: dict = {"top_k": top_k}
        if location is not None:
            conditions.append("metadata->>'location' = :location")
            params["location"] = location
        if country is not None:
            conditions.append("metadata->>'country' = :country")
            params["country"] = country
        if tags:
            conditions.append("metadata->'tags' ?| cast(:tags as text[])")
            params["tags"] = tags

        where_clause = " AND ".join(conditions) if conditions else "TRUE"
        sql_str = f"""
            SELECT id, content, document_id, metadata,
                   1 - (embedding <=> '{vector_str}'::vector) AS similarity_score
            FROM chunks
            WHERE embedding IS NOT NULL AND {where_clause}
            ORDER BY embedding <=> '{vector_str}'::vector
            LIMIT :top_k
            """
        sql = text(sql_str)
        result = await session.execute(sql, params)
        rows = result.mappings().all()

        results = [
            {
                "chunk_id": str(row["id"]),
                "content": row["content"],
                "similarity_score": float(row["similarity_score"]),
                "document_id": str(row["document_id"]),
                "metadata": dict(row["metadata"]) if row["metadata"] else {},
            }
            for row in rows
        ]
        span.set_attribute("search.results_count", len(results))
        if results:
            span.set_attribute("search.top_similarity_score", results[0]["similarity_score"])
        return results
