import logging
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