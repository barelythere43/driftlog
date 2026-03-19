from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

from dateutil import parser as dateutil_parser
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_db, init_db
from src.generation.generator import generate_answer
from src.ingestion.pipeline import process_document
from src.ingestion.transcriber import transcribe_journal_images
import src.models  # noqa: F401 — register models with Base.metadata for init_db
from src.models import Document
from src.schemas import Citation, IngestRequest, IngestResponse, JournalIngestRequest, QueryRequest, QueryResponse
from src.retrieval.dense import dense_search
from src.retrieval.sparse import bm25_index
from src.retrieval.fusion import fuse_results
from src.retrieval.reranker import rerank

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    logger.info("Building BM25 index...")
    await bm25_index.build_index()
    logger.info("BM25 index built.")
    yield


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(
    title="DriftLog",
    description="AI-powered travel knowledge system",
    version="0.1.0",
    lifespan=lifespan,
)

# Serve frontend static assets on /static, and index.html on /
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


def _parse_entry_date(value: str | None) -> date | None:
    if not value or not value.strip():
        return None
    try:
        return dateutil_parser.parse(value.strip()).date()
    except (ValueError, TypeError):
        return None


@app.post("/api/v1/ingest", response_model=IngestResponse)
async def ingest(body: IngestRequest, db: AsyncSession = Depends(get_db)):
    total_chunks = 0
    for doc in body.documents:
        row = Document(
            content=doc.content,
            source=doc.source,
            location=doc.location,
            country=doc.country,
            tags=doc.tags,
            entry_date=_parse_entry_date(doc.entry_date),
        )
        db.add(row)
        total_chunks += await process_document(db, row)
    return IngestResponse(
        job_id=str(uuid.uuid4()),
        status="completed",
        document_count=len(body.documents),
        chunk_count=total_chunks,
    )


@app.post("/api/v1/ingest/journal", response_model=IngestResponse)
async def ingest_journal(body: JournalIngestRequest, db: AsyncSession = Depends(get_db)):
    entries = await transcribe_journal_images([{"data": img.data, "media_type": img.media_type} for img in body.images])
    if not entries or all(not (e.get("transcription") or "").strip() for e in entries):
        raise HTTPException(status_code=422, detail="Could not transcribe any text from the provided images")
    total_chunks = 0
    document_count = 0
    for entry in entries:
        transcription = (entry.get("transcription") or "").strip()
        if not transcription:
            continue
        meta = entry.get("metadata") or {}
        source = body.source
        location = body.location if body.location is not None else meta.get("location")
        country = body.country if body.country is not None else meta.get("country")
        tags = body.tags if body.tags is not None else meta.get("tags") or []
        entry_date = _parse_entry_date(body.entry_date) if body.entry_date is not None else _parse_entry_date(meta.get("date"))
        row = Document(
            content=transcription,
            source=source,
            location=location,
            country=country,
            tags=tags,
            entry_date=entry_date,
        )
        db.add(row)
        total_chunks += await process_document(db, row)
        document_count += 1
    await bm25_index.build_index()
    return IngestResponse(
        job_id=str(uuid.uuid4()),
        status="completed",
        document_count=document_count,
        chunk_count=total_chunks,
    )


@app.post("/api/v1/query", response_model=QueryResponse)
async def query(body: QueryRequest, db: AsyncSession = Depends(get_db)):
    f = body.filters
    dense_chunks = await dense_search(
        db,
        body.question,
        location=f.location if f else None,
        country=f.country if f else None,
        tags=f.tags if f else None,
    )
    sparse_chunks = bm25_index.search(body.question, top_k=20)
    trace_id = str(uuid.uuid4())
    if not dense_chunks and not sparse_chunks:
        return QueryResponse(
            answer="I don't have enough information to answer that.",
            confidence=0.0,
            citations=[],
            query_type="factual",
            retrieval_strategy="hybrid_rrf_rerank",
            chunks_retrieved=0,
            chunks_after_rerank=0,
            trace_id=trace_id,
        )
    fused = fuse_results(dense_chunks, sparse_chunks, top_k=20)
    reranked = await rerank(body.question, fused, top_n=5)
    if not reranked:
        return QueryResponse(
            answer="I don't have enough information to answer that.",
            confidence=0.0,
            citations=[],
            query_type="factual",
            retrieval_strategy="hybrid_rrf_rerank",
            chunks_retrieved=len(fused),
            chunks_after_rerank=0,
            trace_id=trace_id,
        )
    result = await generate_answer(body.question, reranked)
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
        retrieval_strategy="hybrid_rrf_rerank",
        chunks_retrieved=len(fused),
        chunks_after_rerank=len(reranked),
        trace_id=trace_id,
    )