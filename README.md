# Private AI Platform

Self-hosted RAG platform that runs entirely on your own machine. Documents,
embeddings, conversation history and the LLM never leave the host: generation is
served by a local Gemma model through MLX on Apple Silicon.

- **Backend** — FastAPI service (`http://127.0.0.1:8000`) with documents,
  conversations, RAG and a LangGraph [agent layer](#agent-architecture).
- **Celery worker** — ingests uploaded PDFs in the background. Runs **natively**
  next to the backend, because it loads the embedding models.
- **Inference service** — separate FastAPI service (`http://127.0.0.1:8001`)
  wrapping `mlx-community/gemma-3-4b-it-qat-4bit` via MLX-VLM. Runs **natively**
  on the host, never in Docker, because it needs Metal access.
- **PostgreSQL / Redis / Qdrant / RabbitMQ** — run in Docker Compose.

---

## Architecture

```mermaid
flowchart TD
    Client[Client / curl / UI]

    Client -->|HTTP :8000| Backend[Backend FastAPI]

    Backend --> Postgres[(PostgreSQL 16<br/>users, conversations,<br/>messages, documents, chunks)]
    Backend --> Redis[(Redis<br/>cache, Celery results)]
    Backend --> Qdrant[(Qdrant<br/>vector collection 'documents')]
    Backend -->|X-API-Key, :8001| Inference[Inference API FastAPI]
    Backend -->|enqueue document_id| Rabbit[(RabbitMQ<br/>queue 'documents')]
    Backend -->|write PDF| Files[/data/uploads/]

    Rabbit -->|consume| Worker[Celery worker]
    Worker -->|read PDF| Files
    Worker --> Postgres
    Worker --> Qdrant
    Worker --> Redis

    Inference --> MLX[MLX / MLX-VLM<br/>gemma-3-4b-it-qat-4bit]

    subgraph Docker
        Postgres
        Redis
        Qdrant
        Rabbit
    end

    subgraph "Native host (Apple Silicon)"
        Backend
        Worker
        Inference
        MLX
    end
```

### Document ingestion flow

```mermaid
sequenceDiagram
    participant C as Client
    participant B as Backend
    participant F as data/uploads
    participant P as PostgreSQL
    participant R as RabbitMQ
    participant W as Celery worker
    participant Q as Qdrant

    C->>B: POST /documents (multipart PDF)
    B->>B: validate extension, content type, %PDF magic, size
    B->>F: write {document_id}.pdf
    B->>P: INSERT Document(status="processing")
    B->>R: enqueue documents.process(document_id)
    B->>P: UPDATE celery_task_id
    B-->>C: 202 {document_id, status, task_id}

    R->>W: deliver task
    W->>P: check status is still "processing"
    W->>F: read PDF
    W->>W: parse, chunk, embed
    W->>P: guard — document still exists?
    W->>Q: delete old points, upsert deterministic point ids
    W->>P: replace chunks, status="ready"
    W->>F: delete source PDF

    C->>B: GET /documents/{id}
    B-->>C: {"status": "ready", "chunks_count": 48}
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
  prompts/               system prompts (__init__.py, agent.py)
  observability.py       Prometheus metrics + stage/agent logging
  api/                   health.py, documents.py, conversations.py, rag.py
  agent/
    graph.py               LangGraph state machine + AgentService
    state.py               typed, JSON-serialisable AgentState
    nodes.py               classify / direct / rag / tool / fallback / compose
    schemas.py             Pydantic models the LLM must fill in
    structured.py          strict JSON output with one repair attempt
    tools/
      base.py                Tool interface and ToolContext
      registry.py            explicit allowlist, argument validation
      calculator.py          AST-allowlist arithmetic (no eval)
      documents.py           search_documents, get_document_metadata
      datetime_tool.py       get_current_datetime
  services/
    chunking.py            PDF extraction + word-window chunking (pure)
    embeddings.py          e5 encoder + cross-encoder reranker (loaded once)
    vector_store.py        Qdrant access, payload shape, document filtering
    rag_service.py         retrieve -> rerank -> build grounded context
    storage.py             local PDF storage, traversal-safe paths
    document_service.py    validation, CRUD, enqueue, delete
    document_processor.py  the heavy pipeline the worker runs
    conversation_service.py chat turn orchestration
    inference_client.py    pooled httpx client for the inference service
    task_queue.py          TaskDispatcher protocol (no Celery import)
    broker.py              RabbitMQ / Celery liveness probes
  worker/
    celery_app.py          Celery instance, queues, retry defaults, signals
    tasks.py               documents.process task + async bridge
    context.py             per-process resources, model reuse
    dispatch.py            Celery implementation of TaskDispatcher
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

This starts PostgreSQL 16 (`:5432`), Redis (`:6379`), Qdrant (`:6333`) and
RabbitMQ (`:5672`, management UI on `:15672`) — every port bound to `127.0.0.1`
only, so nothing is reachable from outside this machine.

RabbitMQ takes ~20 s to report healthy on a cold start. Check it with:

```bash
docker compose ps
docker exec private-ai-rabbitmq rabbitmq-diagnostics -q ping
```

The management UI is at <http://127.0.0.1:15672> (default dev credentials
`privateai` / `privateai_dev_password`, override with `RABBITMQ_USER` and
`RABBITMQ_PASSWORD`).

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

On startup the backend creates the Qdrant collection if it is missing, prepares
`UPLOAD_DIR` and loads the embedding and reranker models once (set
`PRELOAD_MODELS=false` to defer this until the first RAG request). Interactive
docs: <http://127.0.0.1:8000/docs>.

## 5. Start the Celery worker

In a second terminal, from the repository root:

```bash
celery -A backend.worker.celery_app worker \
  --loglevel=info \
  --pool=solo \
  --queues=documents
```

**Use `--pool=solo` on Apple Silicon.** Celery's default `prefork` pool forks
the process, and forking after torch has initialised Metal/MPS is unsafe — the
child inherits GPU state it does not own and either hangs or crashes. `solo`
runs tasks in the main process, so the embedding and reranker models are loaded
once and reused by every task, which is also what you want for throughput on a
single machine.

`--pool=threads --concurrency=2` also works and keeps one shared copy of the
models; use it if you want a little concurrency. If you ever do run `prefork`,
the models are re-loaded per child (the `worker_process_init` signal
deliberately drops any inherited handles) and you should add
`OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` to the environment.

The worker loads the models lazily, on the first document it processes, so
startup is instant and the first ingestion is the slow one.

Verify the worker is connected:

```bash
curl -s http://127.0.0.1:8000/health/workers | jq
```

```json
{"workers": ["celery@your-mac"], "available": true}
```

---

## API walkthrough

### Health

```bash
curl -s http://127.0.0.1:8000/health | jq
```

```json
{"api":"ok","postgres":"ok","redis":"ok","qdrant":"ok","inference":"ok","rabbitmq":"ok"}
```

`GET /health` never blocks on the worker fleet. Use `GET /health/workers` for
that — it waits out its timeout when nobody answers, which is why it is a
separate route.

### Upload a PDF (asynchronous)

```bash
curl -s -X POST http://127.0.0.1:8000/documents \
  -F "file=@/path/to/report.pdf" | jq
```

```json
{
  "document_id": "0f1a…",
  "status": "processing",
  "task_id": "8f2b…"
}
```

`202 Accepted` comes back as soon as the file is on disk and the job is queued.
The request only does cheap validation — extension, content type, `%PDF` magic
bytes and the size limit — so a file over `MAX_UPLOAD_SIZE_MB` is still rejected
with `413` before anything is queued.

### Poll until it is ready

```bash
curl -s http://127.0.0.1:8000/documents/<document_id> | jq
```

```json
{
  "id": "0f1a…",
  "filename": "report.pdf",
  "status": "ready",
  "total_pages": 12,
  "extracted_pages": 12,
  "chunks_count": 48,
  "error_message": null,
  "celery_task_id": "8f2b…"
}
```

`status` moves `processing → ready` or `processing → failed`. A failed document
keeps a short, safe `error_message` (never a traceback) so you can see what went
wrong; the traceback is in the worker log.

A small polling loop:

```bash
DOC=$(curl -s -X POST http://127.0.0.1:8000/documents \
        -F "file=@/path/to/report.pdf" | jq -r .document_id)

until [ "$(curl -s http://127.0.0.1:8000/documents/$DOC | jq -r .status)" != "processing" ]; do
  sleep 2
done

curl -s http://127.0.0.1:8000/documents/$DOC | jq '{status, chunks_count, error_message}'
```

There is deliberately no separate `GET /documents/{id}/status`: the document
resource already carries the status, the metadata, the task id and the error
message, and it is a cheap single-row read (chunks are only loaded when you ask
for `?include_chunks=true`). A second endpoint would return a subset of the same
data.

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

## Agent Architecture

`POST /conversations/{id}/agent` answers a turn through a LangGraph state
machine that decides *how* to answer before answering. It sits on top of the
services that already exist — it does not replace them, and every endpoint above
keeps working exactly as before.

**It is not an autonomous loop.** The graph is a fixed, acyclic state machine
with four branches that all converge on a single composition step. There is no
edge back to the router, so a run visits exactly three nodes. `AGENT_MAX_STEPS`
(default 6) is a hard ceiling every node checks anyway, so extending the graph
later can never turn it into a runaway.

```mermaid
flowchart TD
    User([User turn]) --> Classify{classify<br/>structured LLM output}

    Classify -->|direct_answer| Direct[LLM only<br/>history, no context]
    Classify -->|rag_search| Rag[RagService:<br/>retrieve → rerank]
    Classify -->|tool| Tools[Tool registry<br/>allowlist]
    Classify -->|unparseable| Fallback[safe fallback]

    Tools --> Calc[calculator]
    Tools --> Search[search_documents]
    Tools --> Meta[get_document_metadata]
    Tools --> Clock[get_current_datetime]

    Direct --> Compose[compose_answer]
    Rag --> Compose
    Calc --> Compose
    Search --> Compose
    Meta --> Compose
    Clock --> Compose
    Fallback --> Compose

    Compose --> Answer([Answer + route + tools + sources])
```

### Routes

| Route | When | What runs |
| --- | --- | --- |
| `direct_answer` | ordinary question or chit-chat | the model, with history and no retrieved context |
| `rag_search` | question about the document base | the existing `RagService` — retrieve, rerank, ground |
| `tool` | arithmetic, clock, document metadata, explicit search | one whitelisted tool, then the model summarises its result |
| `fallback` | routing could not be trusted | a short, honest "please rephrase" answer |

### Tools

There is no `eval`, no dynamic import and no lookup by string. The registry is
an explicit allowlist of instances, and arguments are validated against each
tool's Pydantic schema *before* anything executes.

| Tool | Arguments | Notes |
| --- | --- | --- |
| `calculator` | `expression` | `+ - * / ** %` and parentheses, AST allowlist |
| `search_documents` | `query`, `document_ids?`, `top_k?` | delegates to `RagService`; produces citable sources |
| `get_document_metadata` | `document_id` | reads PostgreSQL |
| `get_current_datetime` | — | UTC, no external API |

The calculator parses with `ast` and walks the tree by hand. Only
`Expression`, `BinOp`, `UnaryOp`, `Constant` and the six arithmetic operators
are accepted; names, calls, attributes, subscripts, comprehensions and literals
of any other type are rejected before evaluation. Expression length, AST depth,
result magnitude and exponent size are all bounded, and a power is rejected on
its estimated magnitude *before* it is computed — `10 ** 100000000` fails
instantly rather than allocating.

### Structured tool calling

The local Gemma build has neither a JSON mode nor native function calling, so
the contract is enforced from our side: a strict system prompt plus the model's
JSON Schema, then the first balanced `{...}` is located in the reply and
validated with Pydantic. An invalid reply gets **one** repair round trip that
shows the model its own output and the validation error. A second failure gives
up — two LLM calls is the hard ceiling, there is no repair loop.

```json
{"route": "tool", "tool_name": "calculator", "reason": "…"}
{"tool": "calculator", "arguments": {"expression": "125 * 8"}}
```

### Fallback behaviour

| Situation | Result |
| --- | --- |
| routing output unparseable after one repair | `rag_search` if `use_rag`, else `direct_answer` |
| router names a tool that is not in the registry | degrade to the safe default route |
| tool selection unparseable after one repair | `fallback` route, honest answer |
| unknown tool / invalid arguments / tool raises | controlled `{success: false, error}`, model explains it |
| retrieval returns nothing | "not enough information in the documents" — never an invented answer |
| step limit exceeded | graph stops, controlled answer, reason logged |
| inference service unreachable | `502` — the turn is not faked with a misleading answer |

### Example

```bash
CONV=$(curl -s -X POST http://127.0.0.1:8000/conversations \
        -H "Content-Type: application/json" \
        -d '{"title": "Агент"}' | jq -r .id)

curl -s -X POST http://127.0.0.1:8000/conversations/$CONV/agent \
  -H "Content-Type: application/json" \
  -d '{"content": "Сколько будет 17 * 23?"}' | jq
```

```json
{
  "message": {
    "id": "…",
    "conversation_id": "…",
    "role": "assistant",
    "content": "17 умножить на 23 равно 391.",
    "created_at": "2026-09-07T12:24:55Z"
  },
  "route": "tool",
  "tools_used": [{"name": "calculator", "success": true, "error": null}],
  "sources": []
}
```

Grounded question over a document:

```bash
curl -s -X POST http://127.0.0.1:8000/conversations/$CONV/agent \
  -H "Content-Type: application/json" \
  -d '{
        "content": "Кто руководит проектом Борей?",
        "use_rag": true,
        "document_ids": ["<document_id>"]
      }' | jq '{route, sources: (.sources | length), answer: .message.content}'
```

```json
{
  "route": "rag_search",
  "sources": 2,
  "answer": "Руководитель проекта Борей — Павел Семёнов … [SOURCE 1]"
}
```

`use_rag` is a hint, not a command: the agent may still route elsewhere, and the
flag decides the safe default when routing cannot be trusted.

### What the response never contains

Only the answer, the chosen route, tool names with success/error metadata, and
sources. Prompts, the router's rationale and any intermediate state stay
server-side. The database keeps plain `user` / `assistant` messages only — the
agent's internal state is never written to a `Message` row.

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
| `AGENT_MAX_STEPS` | `6` | hard ceiling on graph nodes per run |
| `AGENT_ROUTER_TEMPERATURE` | `0.0` | structured calls want determinism |
| `AGENT_STRUCTURED_REPAIR_ATTEMPTS` | `1` | JSON repair round trips; `0` disables |
| `AGENT_STRUCTURED_MAX_TOKENS` | `300` | budget for routing / tool selection |
| `AGENT_ANSWER_MAX_TOKENS` | `600` | budget for the composed answer |
| `MAX_UPLOAD_SIZE_MB` | `25` | upload size limit (`413` above it) |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | encoder |
| `RERANKER_MODEL` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | cross-encoder |
| `EMBEDDING_DIM` | `384` | must match the encoder output size |
| `CHUNK_SIZE_WORDS` / `CHUNK_OVERLAP_WORDS` | `220` / `40` | chunking window |
| `MAX_CONTEXT_CHARS` | `12000` | cap on the grounded context |
| `PRELOAD_MODELS` | `true` | load models at startup, not on first use |
| `UPLOAD_DIR` | `data/uploads` | where PDFs wait for a worker |
| `DELETE_SOURCE_AFTER_PROCESSING` | `true` | drop the PDF once it is indexed |
| `CELERY_BROKER_URL` | `amqp://…@127.0.0.1:5672//` | RabbitMQ connection |
| `CELERY_RESULT_BACKEND` | `redis://127.0.0.1:6379/1` | Celery results (optional) |
| `CELERY_TASK_QUEUE` | `documents` | queue the worker consumes |
| `CELERY_MAX_RETRIES` | `5` | retries for transient failures |
| `CELERY_RETRY_BACKOFF_SECONDS` | `5` | base backoff, doubled with jitter |
| `CELERY_RETRY_BACKOFF_MAX_SECONDS` | `300` | backoff ceiling |
| `CELERY_TASK_SOFT_TIME_LIMIT` / `CELERY_TASK_TIME_LIMIT` | `1500` / `1800` | per-task limits |
| `WORKER_METRICS_PORT` | `0` | worker Prometheus port; `0` disables it |
| `RABBITMQ_USER` / `RABBITMQ_PASSWORD` | dev defaults | consumed by docker-compose |
| `LOG_LEVEL` | `INFO` | logging level |

`.env` is git-ignored, and so is `data/`. The API key, prompt bodies and
document text are never written to the logs — processing logs carry the
document id, the task id, the stage name, a duration and counts only.

## Observability

Structured logs per ingestion stage (`load`, `read_source`, `parse`, `chunk`,
`embed`, `index`, `persist`, `cleanup_source`):

```
document_stage stage=embed document_id=0f1a… task_id=8f2b… duration_ms=4180.2 vectors=48
```

Agent runs log the same way — identifiers, route, tool name, step and duration,
never the user's message or the prompts:

```
agent_started   conversation_id=30d9c013… history=1 scoped=False use_rag=False
agent_routed    conversation_id=30d9c013… duration_ms=2734.4 route=tool step=1 tool_name=calculator
agent_tool_completed conversation_id=30d9c013… duration_ms=2085.1 step=2 success=True tool_name=calculator
agent_completed conversation_id=30d9c013… duration_ms=5697.3 errors=0 route=tool sources=0 step=3 tools=1
```

Prometheus metrics — `documents_processing_total`,
`document_processing_failures_total{reason}`,
`document_processing_duration_seconds{outcome}`,
`document_processing_stage_seconds{stage}`, plus
`agent_requests_total{route,status}`, `agent_duration_seconds` and
`agent_tool_calls_total{tool,status}`:

```bash
curl -s http://127.0.0.1:8000/metrics | grep document_processing
```

**Known limitation.** The API process and the Celery worker are separate
processes, so `GET /metrics` on the backend does not include worker counters —
and it is the worker that does the processing. Set `WORKER_METRICS_PORT=9100`
to have the worker serve its own `/metrics`, and scrape both targets. That is
correct for a single-process pool (`--pool=solo` or `--pool=threads`), which is
the recommended setup here. Aggregating a multi-child `prefork` pool into one
endpoint needs `prometheus_client`'s multiprocess mode and a shared
`PROMETHEUS_MULTIPROC_DIR`; rather than ship a version that silently reports
only the parent's numbers, that is left as a TODO.

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
| `202` | upload accepted and queued (not an error, but not `201` either) |
| `400` | wrong extension or content type, missing `%PDF` magic, empty file |
| `404` | unknown conversation or document |
| `413` | upload above `MAX_UPLOAD_SIZE_MB` |
| `422` | request body or path parameter failed validation |
| `502` | inference service unreachable or returned an error |
| `503` | Qdrant unavailable during a delete |

Since ingestion moved to a worker, a PDF that is well-formed enough to pass the
magic-byte check but cannot actually be parsed is no longer a `400` on
`POST /documents`. It is accepted, and the failure shows up as
`status: "failed"` with an `error_message` on the document.

## Development

```bash
python -m compileall backend     # syntax check
ruff check backend tests         # lint
pytest -q                        # offline unit tests
pytest -m integration            # needs live Postgres, Qdrant and RabbitMQ
alembic check                    # models and migrations agree
```

Unit tests never touch the network: the models, Qdrant, Redis, RabbitMQ and the
inference service are replaced by in-process fakes, and PostgreSQL by in-memory
SQLite. The Celery task body is tested by calling it directly with those fakes,
and its retry policy in Celery's eager mode — no broker is needed. Agent tests
script the fake inference client's replies, so routing, tool calling and every
fallback path run without a model.

The agent integration tests drive the live HTTP API (routing runs on the real
local Gemma). They skip themselves when the backend or the inference service is
not up, and `BACKEND_URL` overrides the target:

```bash
pytest -m integration -o addopts=""
```

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
- **No progress detail.** A document is `processing` or it is not; there is no
  percentage, no stage readout and no push notification — poll the document.
- **Deep PDF validation is deferred.** `POST /documents` cannot tell you a PDF
  is unparseable, because it does not parse it. You learn that from the
  document's `status` afterwards.
- **Worker metrics are a separate scrape target.** See
  [Observability](#observability); a merged multiprocess registry is a TODO.
- **Revoking is best effort.** `DELETE` on a processing document revokes the
  task, but a worker that already started keeps running until its next guard
  check. Correctness does not depend on the revoke — the guards do.
- **Scanned PDFs produce nothing.** There is no OCR, so image-only pages are
  skipped and a fully scanned document ends up `failed`.
- **The deprecated `POST /documents/upload` still blocks.** It is kept for
  backward compatibility and runs the pipeline inline; it does not use the
  worker at all.
- **Single inference process.** The inference service serialises generation
  behind a lock, so concurrent chat requests queue up.
- **No streaming through the backend.** The inference service exposes
  `/v1/chat/stream`, but the backend only uses `/v1/chat`.
- **Redis is only health-checked.** It is running and wired up but not yet used
  for caching or rate limiting.
- **Deleting a document does not rewrite history.** Past assistant messages keep
  the answers that were grounded on it.
- **The agent takes one tool per turn.** The graph is a fixed DAG, so it cannot
  chain a search into a calculation. Multi-step plans would need a new node and
  an edge back into the tool step — and that is exactly what `AGENT_MAX_STEPS`
  is there to bound.
- **Routing quality is bounded by a 4B local model.** It is right on clear-cut
  questions (see the smoke tests) but can misroute ambiguous ones. Every
  misroute degrades to a safe branch rather than failing.
- **Routing costs an extra LLM round trip.** An agent turn is one call slower
  than `POST /conversations/{id}/messages`; on this hardware the router adds
  ~3 s. Use the plain endpoint when you already know you want RAG.
- **`langchain-core` comes along with LangGraph.** It is a hard dependency of
  `langgraph`, not a choice; no LangChain chains, agents or LLM wrappers are
  used. `websockets` is pinned to `16.1.1` because `langgraph-sdk` requires
  `<17`; the project has no WebSocket routes, so nothing depends on the newer
  release.
- **No agent tracing UI.** LangSmith is installed transitively but not
  configured, and no checkpointer is attached — the graph is stateless between
  turns and memory comes from the `messages` table.
