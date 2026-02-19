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

## Day 3
- **Retrieval:** `src/retrieval/dense.py` — `dense_search` (embed query via `embed_text`, pgvector cosine `<=>`, optional filters: location, country, tags with JSONB/array). Vector passed as string literal in SQL (`'{vector_str}'::vector`) to avoid async driver serialization issues.
- **Generation:** `src/generation/prompts.py` — `build_prompt(question, chunks)` returns `{system, user}`; travel assistant rules, cited context blocks. `src/generation/generator.py` — `generate_answer(question, chunks)` calls Claude (claude-3-5-sonnet-20241022), parses `[1]` citations, builds citations list with excerpt, confidence = avg similarity of cited chunks; cached `AsyncAnthropic` client.
- **Query API:** POST /api/v1/query — `QueryRequest` (question + optional `filters`: QueryFilters with location, country, tags, date_range), `QueryResponse` (answer, confidence, citations, query_type, retrieval_strategy, chunks_retrieved, chunks_after_rerank, trace_id). Citation model: index, chunk_id, source, location, excerpt. Short-circuit when no chunks (low-confidence refusal, no Claude call); trace_id = UUID for now.
- **Schemas:** QueryFilters, DateRange, QueryRequest (question + filters), QueryResponse, Citation.
- **Chunk index:** Switched embedding index from IVFFlat to HNSW with `vector_cosine_ops`; comment noting HNSW preferred for small-to-medium datasets (IVFFlat needs list/probe tuning).
