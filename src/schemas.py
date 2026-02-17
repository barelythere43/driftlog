from pydantic import BaseModel


class IngestDocumentRequest(BaseModel):
    content: str
    source: str | None = None
    location: str | None = None
    country: str | None = None
    tags: list[str] | None = None


class IngestRequest(BaseModel):
    documents: list[IngestDocumentRequest]


class IngestResponse(BaseModel):
    job_id: str
    status: str
    document_count: int


class HealthResponse(BaseModel):
    status: str
    version: str
