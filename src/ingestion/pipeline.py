from sqlalchemy.ext.asyncio import AsyncSession

from src.ingestion.chunker import chunk_text
from src.ingestion.embedder import embed_chunks
from src.models import Chunk, Document
from src.tracing import get_tracer, timed_span


async def process_document(session: AsyncSession, document: Document) -> int:
    """Chunk document content, embed chunks, and persist Chunk rows with metadata. Returns number of chunks created."""
    tracer = get_tracer()
    with timed_span(tracer, "ingestion.process_document", {
        "document.source": document.source or "",
        "document.location": document.location or "",
        "document.country": document.country or "",
    }) as span:
        await session.flush()  # ensure document.id is set if document was just added
        span.set_attribute("document.id", str(document.id))

        with timed_span(tracer, "ingestion.chunking") as chunk_span:
            chunks = chunk_text(document.content)
            chunk_span.set_attribute("chunking.input_length", len(document.content))
            chunk_span.set_attribute("chunking.chunk_count", len(chunks))

        if not chunks:
            span.set_attribute("ingestion.chunks_created", 0)
            return 0

        embeddings = await embed_chunks(chunks)

        metadata = {
            "source": document.source,
            "location": document.location,
            "country": document.country,
            "tags": document.tags,
            "entry_date": document.entry_date.isoformat() if document.entry_date else None,
        }

        with timed_span(tracer, "ingestion.persist_chunks") as persist_span:
            for i, content in enumerate(chunks):
                embedding = embeddings[i] if i < len(embeddings) else None
                chunk = Chunk(
                    document_id=document.id,
                    content=content,
                    chunk_index=i,
                    embedding=embedding,
                    metadata_=metadata,
                )
                session.add(chunk)
            persist_span.set_attribute("persist.chunk_count", len(chunks))

        span.set_attribute("ingestion.chunks_created", len(chunks))
        return len(chunks)
