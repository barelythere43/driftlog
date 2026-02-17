# driftlog

## Day 1
- Created repo and project structure
- FastAPI skeleton with /health endpoint
- Dockerfile with layer caching + non-root user
- docker-compose with PostgreSQL/pgvector + health checks
- Verified: API returns 200, Swagger UI works

## Day 2
- **Database:** Async SQLAlchemy engine + session factory, `get_db` dependency, `init_db` with pgvector extension creation on startup
- **Models:** `Document` (content, source, location, country, tags, timestamps) and `Chunk` (document_id FK, content, chunk_index, embedding Vector(1536), metadata_ JSONB) with IVFFlat/GIN/btree indexes
- **Schemas:** IngestDocumentRequest, IngestRequest, IngestResponse, HealthResponse (Pydantic)
- **Ingest API:** POST /api/v1/ingest accepts documents, persists Document rows, runs chunk → embed → Chunk pipeline per doc
- **Ingestion pipeline:** `chunk_text` (RecursiveCharacterTextSplitter 800/200), `embed_chunks` (OpenAI text-embedding-3-small, batched), `process_document` (chunk → embed → save Chunks with doc metadata)
- **Scripts:** scripts/test_embedder.py (loads .env, path hack, prints embedding count + dim)
- **Docker:** Copy scripts/ into image; added env_file for .env so OPENAI_API_KEY available in container
- **Config:** requires-python relaxed to >=3.10
- **.gitignore:** Python caches, venvs, egg-info, .env (keep .env.example), editors, OS junk