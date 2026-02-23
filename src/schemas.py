from pydantic import BaseModel, Field


class JournalImage(BaseModel):
    data: str
    media_type: str


class JournalIngestRequest(BaseModel):
    images: list[JournalImage] = Field(..., min_length=1)
    location: str | None = None
    country: str | None = None
    tags: list[str] | None = None
    source: str = "journal"
    entry_date: str | None = None


class IngestDocumentRequest(BaseModel):
    content: str
    source: str | None = None
    location: str | None = None
    country: str | None = None
    tags: list[str] | None = None
    entry_date: str | None = None


class IngestRequest(BaseModel):
    documents: list[IngestDocumentRequest]


class IngestResponse(BaseModel):
    job_id: str
    status: str
    document_count: int
    chunk_count: int = 0


class HealthResponse(BaseModel):
    status: str
    version: str


class Citation(BaseModel):
    index: int
    chunk_id: str
    source: str
    location: str
    excerpt: str


class DateRange(BaseModel):
    start: str | None = None
    end: str | None = None


class QueryFilters(BaseModel):
    location: str | None = None
    country: str | None = None
    tags: list[str] | None = None
    date_range: DateRange | None = None


class QueryRequest(BaseModel):
    question: str
    filters: QueryFilters | None = None


class QueryResponse(BaseModel):
    answer: str
    confidence: float
    citations: list[Citation]
    query_type: str
    retrieval_strategy: str
    chunks_retrieved: int
    chunks_after_rerank: int
    trace_id: str
