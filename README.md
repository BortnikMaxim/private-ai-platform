# Private AI Platform

Self-hosted RAG platform that runs entirely on your own machine. Documents,
embeddings, conversation history and the LLM never leave the host: generation is
served by a local Gemma model through MLX on Apple Silicon.

- **Backend** — FastAPI service (`http://127.0.0.1:8000`) with documents,
  conversations and RAG APIs.
- **Inference service** — separate FastAPI service (`http://127.0.0.1:8001`)
  wrapping `mlx-community/gemma-3-4b-it-qat-4bit` via MLX-VLM. Runs **natively**
  on the host, never in Docker, because it needs Metal access.
- **PostgreSQL / Redis / Qdrant** — run in Docker Compose.

---

## Architecture

```mermaid
flowchart TD
    Client[Client / curl / UI]

    Client -->|HTTP :8000| Backend[Backend FastAPI]

    Backend --> Postgres[(PostgreSQL 16<br/>users, conversations,<br/>messages, documents, chunks)]
    Backend --> Redis[(Redis<br/>cache / health)]
    Backend --> Qdrant[(Qdrant<br/>vector collection 'documents')]
    Backend -->|X-API-Key, :8001| Inference[Inference API FastAPI]

    Inference --> MLX[MLX / MLX-VLM<br/>gemma-3-4b-it-qat-4bit]

    subgraph Docker
        Postgres
        Redis
        Qdrant
    end

    subgraph "Native host (Apple Silicon)"
        Inference
        MLX
    end
```

### Request flow for a grounded answer

```mermaid
sequenceDiagram
    participant C as Client
    participant B as Backend
    participant Q as Qdrant
    participant P as PostgreSQL
    participant I as Inference

    C->>B: POST /conversations/{id}/messages (use_rag=true)
    B->>P: save user message
    B->>Q: vector search (top candidate_k, filtered by document_ids)
    B->>B: cross-encoder rerank -> top_k
    B->>I: POST /v1/chat (system prompt + context + bounded history)
    I-->>B: assistant answer
    B->>P: save assistant message
    B-->>C: assistant message + sources
```

### Code layout

```
backend/
  app.py                 FastAPI app: lifespan, exception handlers, routers
  config.py              pydantic-settings configuration
  db.py                  async engine, session factory, get_db dependency
  models.py              SQLAlchemy 2 models
  schemas.py             request/response models
  dependencies.py        DI providers backed by app.state
  errors.py              domain errors mapped to HTTP status codes
  prompts.py             system prompts
  api/                   health.py, documents.py, conversations.py, rag.py
  services/
    chunking.py            PDF extraction + word-window chunking (pure)
    embeddings.py          e5 encoder + cross-encoder reranker (loaded once)
    vector_store.py        Qdrant access, payload shape, document filtering
    rag_service.py         retrieve -> rerank -> build grounded context
    document_service.py    upload validation and ingestion pipeline
    conversation_service.py chat turn orchestration
    inference_client.py    pooled httpx client for the inference service
  scripts/
    reset_qdrant.py        manual maintenance for the vector collection
inference/               MLX/Gemma service (unchanged)
migrations/              Alembic (async)
tests/                   offline unit tests + opt-in integration tests
```

---

## Requirements

- macOS on Apple Silicon (for the MLX inference service)
- Python 3.11
- Docker Desktop

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-dev.txt   # tests and linting

cp .env.example .env
# then edit .env — at minimum set INFERENCE_API_KEY
```

Generate an API key:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

The same value must be given to both services: `INFERENCE_API_KEY` in `.env`
(read by the backend) and in the environment of the inference service.

## 1. Start the infrastructure

```bash
docker compose up -d
docker compose ps
```

This starts PostgreSQL 16 (`:5432`), Redis (`:6379`) and Qdrant (`:6333`), all
bound to `127.0.0.1` only.

## 2. Run migrations

```bash
alembic upgrade head
```

The migration environment reads `DATABASE_URL` from your settings, so there is
no database URL committed in `alembic.ini`. To target another database ad hoc:

```bash
alembic -x db_url=postgresql+asyncpg://user:pass@host:5432/db upgrade head
```

### Upgrading a database that predates Alembic

Early versions of this project created the schema with `Base.metadata.create_all`
via a since-removed `init_db.py`. Such a database already has `conversations`
and `messages` in their old shape (no `user_id`, no `updated_at`) and no
`alembic_version`, so `alembic upgrade head` fails with *table already exists*.

The initial migration is deliberately a plain baseline — it contains no
workarounds for that older schema. Reset the two legacy tables instead:

```bash
# 1. Back up whatever is in there. --column-inserts is what makes the dump
#    replayable against the new schema; a plain dump also carries the OLD
#    CREATE TABLE statements and will not restore.
docker exec private-ai-postgres pg_dump -U privateai -d privateai \
  -t conversations -t messages --data-only --column-inserts \
  > legacy-conversations.sql

# 2. DESTRUCTIVE: drops the old tables and every row in them.
docker exec private-ai-postgres psql -U privateai -d privateai \
  -c "DROP TABLE IF EXISTS messages CASCADE; DROP TABLE IF EXISTS conversations CASCADE;"

# 3. Build the real schema.
alembic upgrade head

# 4. Optional: replay the old rows. user_id stays NULL and updated_at takes its
#    default, so the dump is compatible with the new columns as-is.
docker exec -i private-ai-postgres psql -U privateai -d privateai \
  < legacy-conversations.sql
```

Vectors already in Qdrant are *not* covered by this: they keep pointing at
document ids that no longer have a row. See
[Maintaining the vector collection](#maintaining-the-vector-collection).

## 3. Start the inference service (native, not Docker)

```bash
INFERENCE_API_KEY=<your-key> \
  uvicorn inference.app:app --host 127.0.0.1 --port 8001
```

The first run downloads the Gemma weights. Check it:

```bash
curl http://127.0.0.1:8001/health
```

## 4. Start the backend

```bash
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

On startup the backend creates the Qdrant collection if it is missing and loads
the embedding and reranker models once (set `PRELOAD_MODELS=false` to defer this
until the first RAG request). Interactive docs: <http://127.0.0.1:8000/docs>.

---

## API walkthrough

### Health

```bash
curl -s http://127.0.0.1:8000/health | jq
```

```json
{"api":"ok","postgres":"ok","redis":"ok","qdrant":"ok","inference":"ok"}
```

### Upload a PDF

```bash
curl -s -X POST http://127.0.0.1:8000/documents \
  -F "file=@/path/to/report.pdf" | jq
```

```json
{
  "id": "0f1a...",
  "filename": "report.pdf",
  "status": "ready",
  "total_pages": 12,
  "extracted_pages": 12,
  "chunks_count": 48,
  "error_message": null
}
```

Ingestion parses the PDF, chunks it, embeds the chunks, writes chunk metadata to
PostgreSQL and vectors plus payload to Qdrant. On failure the document is kept
with `status: "failed"` and an `error_message` so you can see what happened.

List, inspect and delete:

```bash
curl -s http://127.0.0.1:8000/documents | jq
curl -s "http://127.0.0.1:8000/documents/<document_id>?include_chunks=true" | jq
curl -s -X DELETE http://127.0.0.1:8000/documents/<document_id> | jq
```

Deleting a document removes its chunk rows **and** its Qdrant points.

### Create a conversation

```bash
curl -s -X POST http://127.0.0.1:8000/conversations \
  -H "Content-Type: application/json" \
  -d '{"title": "Отчёты за квартал"}' | jq
```

### Ask a grounded question inside a conversation

```bash
curl -s -X POST http://127.0.0.1:8000/conversations/<conversation_id>/messages \
  -H "Content-Type: application/json" \
  -d '{
        "content": "Какие проекты описаны в документе?",
        "use_rag": true,
        "document_ids": ["<document_id>"]
      }' | jq
```

```json
{
  "message": {
    "id": "…",
    "conversation_id": "…",
    "role": "assistant",
    "content": "В документе описаны проекты …",
    "created_at": "2026-09-07T10:00:00Z"
  },
  "used_rag": true,
  "sources": [
    {
      "document_id": "…",
      "filename": "report.pdf",
      "page": 3,
      "chunk_index": 11,
      "vector_score": 0.8412,
      "rerank_score": 6.13,
      "score": 0.8412
    }
  ]
}
```

`document_ids` is optional; when present, retrieval searches **only** those
documents. Omit `use_rag` (or set it to `false`) for an ordinary chat turn.
Conversation history lives in PostgreSQL and survives a backend restart; only
the newest `CHAT_HISTORY_LIMIT` messages are replayed to the model.

Read the transcript back:

```bash
curl -s http://127.0.0.1:8000/conversations/<conversation_id> | jq
curl -s -X DELETE http://127.0.0.1:8000/conversations/<conversation_id> | jq
```

### Stateless RAG endpoints

```bash
# One-shot grounded question, no conversation stored
curl -s -X POST http://127.0.0.1:8000/rag/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Какие проекты описаны?", "top_k": 5}' | jq

# Debug: compare vector hits against reranked results
curl -s -X POST http://127.0.0.1:8000/rag/retrieve \
  -H "Content-Type: application/json" \
  -d '{"question": "Какие проекты описаны?", "top_k": 5, "candidate_k": 15}' | jq
```

---

## Configuration

All settings come from environment variables (or `.env`); see `.env.example`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | local dev Postgres | async SQLAlchemy DSN |
| `REDIS_URL` | `redis://127.0.0.1:6379/0` | Redis connection |
| `QDRANT_URL` | `http://127.0.0.1:6333` | Qdrant connection |
| `QDRANT_COLLECTION` | `documents` | vector collection name |
| `INFERENCE_URL` | `http://127.0.0.1:8001` | inference service base URL |
| `INFERENCE_API_KEY` | *(empty)* | sent as `X-API-Key`; must be set |
| `RAG_TOP_K` | `5` | chunks kept after reranking |
| `RAG_CANDIDATE_K` | `15` | chunks fetched from Qdrant before reranking |
| `CHAT_HISTORY_LIMIT` | `20` | messages replayed to the model |
| `MAX_UPLOAD_SIZE_MB` | `25` | upload size limit (`413` above it) |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | encoder |
| `RERANKER_MODEL` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | cross-encoder |
| `EMBEDDING_DIM` | `384` | must match the encoder output size |
| `CHUNK_SIZE_WORDS` / `CHUNK_OVERLAP_WORDS` | `220` / `40` | chunking window |
| `MAX_CONTEXT_CHARS` | `12000` | cap on the grounded context |
| `PRELOAD_MODELS` | `true` | load models at startup, not on first use |
| `LOG_LEVEL` | `INFO` | logging level |

`.env` is git-ignored. The API key, prompt bodies and document text are never
written to the logs.

## Maintaining the vector collection

Qdrant and PostgreSQL can drift apart — vectors ingested before a schema reset
survive as *orphans* whose `document_id` no longer matches any row, and they
keep turning up as sources in RAG answers.

`backend.scripts.reset_qdrant` is a manually invoked tool for this. It never
runs on startup, it only ever touches the collection named by
`QDRANT_COLLECTION`, and every destructive mode needs an explicit `--yes`.

```bash
# Report only — the default, changes nothing
python -m backend.scripts.reset_qdrant

# Delete only vectors whose document_id has no row in PostgreSQL
python -m backend.scripts.reset_qdrant --purge-orphans --yes

# Drop the collection and recreate it empty with the configured vector size
python -m backend.scripts.reset_qdrant --recreate --yes
```

Sample report:

```
Qdrant     : http://127.0.0.1:6333
Collection : documents
Points     : 9
Documents  : 1 in Qdrant, 0 in PostgreSQL
Orphans    : 1 document(s), 9 point(s) with no Document row
  - 4d1f…: 9 point(s)

Nothing changed. Pass --purge-orphans or --recreate (with --yes).
```

Without `--yes` a destructive mode prints what it would do and exits non-zero.
`--purge-orphans` refuses to run at all when the `documents` table is missing,
since every vector would then look orphaned.

## Error responses

Errors are returned as `{"detail": "..."}` — never a Python traceback.

| Status | Meaning |
| --- | --- |
| `400` | not a PDF, corrupt PDF, or no extractable text |
| `404` | unknown conversation or document |
| `413` | upload above `MAX_UPLOAD_SIZE_MB` |
| `422` | request body failed validation |
| `502` | inference service unreachable or returned an error |
| `503` | Qdrant unavailable during a delete |

## Development

```bash
python -m compileall backend     # syntax check
ruff check backend tests         # lint
pytest -q                        # offline unit tests
pytest -m integration            # needs live Postgres + Qdrant
```

Unit tests never touch the network: the models, Qdrant, Redis and the inference
service are replaced by in-process fakes, and PostgreSQL by in-memory SQLite.

New migration after changing `backend/models.py`:

```bash
alembic revision --autogenerate -m "describe the change"
alembic upgrade head
```

## Current limitations

- **PDF only.** Other formats are rejected at upload.
- **No authentication.** The `users` table exists and `Conversation.user_id` is
  nullable, but there is no login flow yet and the API is unauthenticated —
  keep it bound to `127.0.0.1`.
- **Ingestion is synchronous.** A large PDF blocks its request for the whole
  parse/embed cycle; there is no background worker or progress polling yet, so
  `status: "processing"` is only ever observed on a failed request.
- **Scanned PDFs produce nothing.** There is no OCR, so image-only pages are
  skipped and a fully scanned document fails with `400`.
- **Single inference process.** The inference service serialises generation
  behind a lock, so concurrent chat requests queue up.
- **No streaming through the backend.** The inference service exposes
  `/v1/chat/stream`, but the backend only uses `/v1/chat`.
- **Redis is only health-checked.** It is running and wired up but not yet used
  for caching or rate limiting.
- **Deleting a document does not rewrite history.** Past assistant messages keep
  the answers that were grounded on it.
