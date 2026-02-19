import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_db, init_db
from src.generation.generator import generate_answer
from src.ingestion.pipeline import process_document
import src.models  # noqa: F401 — register models with Base.metadata for init_db
from src.models import Document
from src.schemas import Citation, IngestRequest, IngestResponse, QueryRequest, QueryResponse
from src.retrieval.dense import dense_search


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="DriftLog",
    description="AI-powered travel knowledge system",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/api/v1/ingest", response_model=IngestResponse)
async def ingest(body: IngestRequest, db: AsyncSession = Depends(get_db)):
    for doc in body.documents:
        row = Document(
            content=doc.content,
            source=doc.source,
            location=doc.location,
            country=doc.country,
            tags=doc.tags,
        )
        db.add(row)
        await process_document(db, row)
    return IngestResponse(
        job_id=str(uuid.uuid4()),
        status="completed",
        document_count=len(body.documents),
    )


@app.post("/api/v1/query", response_model=QueryResponse)
async def query(body: QueryRequest, db: AsyncSession = Depends(get_db)):
    f = body.filters
    chunks = await dense_search(
        db,
        body.question,
        location=f.location if f else None,
        country=f.country if f else None,
        tags=f.tags if f else None,
    )
    trace_id = str(uuid.uuid4())
    if not chunks:
        return QueryResponse(
            answer="I don't have enough information to answer that.",
            confidence=0.0,
            citations=[],
            query_type="factual",
            retrieval_strategy="dense_only",
            chunks_retrieved=0,
            chunks_after_rerank=0,
            trace_id=trace_id,
        )
    result = await generate_answer(body.question, chunks)
    citations = [
        Citation(
            index=i,
            chunk_id=c["chunk_id"],
            source=c["source"],
            location=c["location"],
            excerpt=c["excerpt"],
        )
        for i, c in enumerate(result["citations"], start=1)
    ]
    return QueryResponse(
        answer=result["answer"],
        confidence=result["confidence"],
        citations=citations,
        query_type=result["query_type"],
        retrieval_strategy="dense_only",
        chunks_retrieved=result["chunks_retrieved"],
        chunks_after_rerank=result["chunks_after_rerank"],
        trace_id=trace_id,
    )