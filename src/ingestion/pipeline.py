from sqlalchemy.ext.asyncio import AsyncSession

from src.ingestion.chunker import chunk_text
from src.ingestion.embedder import embed_chunks
from src.models import Chunk, Document


async def process_document(session: AsyncSession, document: Document) -> int:
    """Chunk document content, embed chunks, and persist Chunk rows with metadata. Returns number of chunks created."""
    await session.flush()  # ensure document.id is set if document was just added
    chunks = chunk_text(document.content)
    if not chunks:
        return 0

    embeddings = await embed_chunks(chunks)

    metadata = {
        "source": document.source,
        "location": document.location,
        "country": document.country,
        "tags": document.tags,
        "entry_date": document.entry_date.isoformat() if document.entry_date else None,
    }

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
    return len(chunks)
