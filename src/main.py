import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_db, init_db
from src.ingestion.pipeline import process_document
import src.models  # noqa: F401 — register models with Base.metadata for init_db
from src.models import Document
from src.schemas import IngestRequest, IngestResponse


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