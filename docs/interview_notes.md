# Interview notes

Working notes for defending the design of this repository in a technical
interview. Not marketing copy — where a decision is weak or a result is
negative, that is stated plainly, because a reviewer will find it anyway and
the honest version is the defensible one.

**Framing.** This is an independent portfolio project. It has no users, no
production traffic and no operational history. Every claim below is about code
in this repository, not about commercial experience.

---

## The single most important result

**Hybrid retrieval did not beat dense retrieval on the evaluation corpus.**

| config | MRR | Recall@5 | p50 |
| --- | --- | --- | --- |
| dense | 0.888 | 0.883 | 15.5 ms |
| dense+rerank | 0.978 | 0.983 | 111.9 ms |
| hybrid | 0.879 | 0.883 | 18.7 ms |
| hybrid+rerank | 0.978 | 0.983 | 114.3 ms |

Hybrid alone is −0.009 MRR and unchanged Recall@5. Only 4 of 30 queries move,
and they cancel: BM25 rescues q05 (0.500→1.000) and q17 (0.000→0.167), and
breaks q02 (1.000→0.200) and q20 (0.143→0.000).

**Why this is not evidence that hybrid retrieval is worse in general.** The
corpus has 18 documents and 18 chunks. Dense retrieval already scores Recall@5
= 1.000 on the `literal` and `keyword` categories — exactly the categories BM25
exists to rescue. There is no headroom left for the lexical branch to recover,
while its false positives still cost something. Lexical retrieval earns its
keep when the corpus is large enough that embeddings start missing rare literal
tokens: product codes, error identifiers, legal article numbers, surnames. An
18-document set cannot exhibit that failure mode.

**So why keep it on by default?** Because it costs ~3 ms, is metric-neutral
once reranking is on (`hybrid+rerank` and `dense+rerank` are identical on every
metric), and covers a failure mode the corpus is too small to show.
`RETRIEVAL_MODE=dense` restores the previous pipeline exactly.

*Q: "Your own data says the feature does nothing. Why ship it?"*
A: The data says it does nothing **on this corpus**, and I say so in the README
rather than quoting a number that flatters it. It is on because it is free and
insures against a known failure mode; the moment I had a bigger corpus I would
re-run the harness and decide again on evidence.

*Q: "How would you prove it helps?"*
A: Grow the corpus to a few hundred documents with lexically-near duplicates,
and add a query category of rare literal identifiers that appear in exactly one
document. That is where dense recall degrades and RRF starts paying.

---

## Retrieval

### E5 embeddings (`intfloat/multilingual-e5-small`)

**Problem.** The corpus is Russian and English, often in the same request. A
monolingual English model would fail on half of it.

**Why this one.** Multilingual, 384 dimensions, small enough to run on a laptop
alongside a 4B LLM, and it uses the `query:` / `passage:` instruction prefixes —
which the code honours in `EmbeddingService.embed_query` and `embed_passages`.
Using the wrong prefix, or none, measurably degrades E5.

**Alternatives.** `multilingual-e5-base`/`large` (better, several times the
memory and latency), `LaBSE` (strong cross-lingual alignment, heavier),
OpenAI `text-embedding-3` (better quality, but the entire point of this project
is that nothing leaves the machine).

**Trade-off.** 384-dim small model buys speed and memory at some quality. The
cross-encoder recovers most of the gap.

*Q: "Why not just use a bigger embedding model?"*
A: Because the reranker buys more quality per millisecond here. Measured: the
cross-encoder adds +0.090 MRR for ~95 ms. I would spend the memory budget on a
larger embedding model only after the reranker stopped helping.

*Q: "What does the `query:` prefix do?"*
A: E5 is trained with asymmetric instruction prefixes; queries and passages are
embedded into the same space but tagged differently. Dropping them costs recall
because inference no longer matches the training distribution.

### Qdrant

**Problem.** Approximate nearest-neighbour search with server-side payload
filtering, because tenant isolation must happen *inside* the search.

**Why.** Native payload filters with indexes, an async Python client, easy to
run in Docker, and filtering is applied during HNSW traversal rather than after.

**Alternatives.** pgvector (one less service, weaker filtering ergonomics at
scale), FAISS (a library, not a service — no persistence or filtering),
Milvus/Weaviate (more capable, heavier to operate for a single-machine project).

**Limitation.** HNSW is approximate. On 18 points that is invisible; at scale
recall depends on `ef` tuning, which this project never had to do.

### BM25 in Python over `document_chunks`

**Problem.** Add a lexical branch without a second copy of the corpus or a new
service.

**Why this shape.** The chunk text is already in PostgreSQL with the tenant
reachable through `documents.user_id`, so BM25 over those rows needs no new
infrastructure. Okapi BM25 is ~40 lines, exact and inspectable.

**Alternatives.** PostgreSQL full-text search with `ts_rank` (free, but it is a
different ranking function than the one I would be claiming), OpenSearch
(correct at scale, a whole extra service), Qdrant sparse vectors / miniCOIL
(keeps one store, but re-indexing and more moving parts).

**Trade-off, stated in the module docstring and the README.** The index is
in-process, per tenant, held in memory and rebuilt when the corpus changes. That
suits a personal corpus and is wrong for a large multi-tenant deployment.
`LexicalRetriever` is the seam where the swap happens.

*Q: "This does not scale. Why did you build it this way?"*
A: Deliberately, and it is documented as a limitation rather than hidden. For a
self-hosted corpus of a few thousand chunks the rebuild is milliseconds and the
memory is trivial. If it needed to scale I would move the branch behind the same
interface to Postgres FTS or OpenSearch — the call sites would not change.

*Q: "How do you know the cache is not stale?"*
A: A version probe — chunk count plus newest chunk timestamp, one indexed query
— runs before every search. An ingest raises both, a delete lowers the count, a
reprocess replaces timestamps. Any corpus change moves the tuple and forces a
rebuild, which is how a Celery worker's writes become visible to the API
process.

### Russian stemming (Snowball)

**Problem.** I found this on live logs: the lexical branch returned zero hits
for real queries. "проекты" does not match a chunk containing "проект".

**Why it matters.** Russian is heavily inflected. Unstemmed BM25 on a Russian
corpus is decorative — it looks like a feature and contributes nothing.

**Implementation.** Snowball, with the language chosen **per token by script**,
so a bilingual chunk works without a language-detection pass. Digits and
hyphenated identifiers (`INC-2026-017`) are left untouched so they stay exact.

**Alternatives.** pymorphy (better Russian lemmatisation, heavier, unmaintained
for a while), no stemming (measured to be useless here), a language classifier
per document (more machinery for the same result on mixed text).

*Q: "Why not lemmatisation?"*
A: Stemming is cruder and occasionally over-collapses, but it is one small pure
Python dependency and it fixed the actual observed failure. Lemmatisation would
be the next step if I saw stemming-induced false positives in the evaluation —
I have not.

### Reciprocal Rank Fusion

**Problem.** Combine two ranked lists whose scores are not comparable.

**Why RRF.** Cosine similarity and BM25 live on different, unbounded,
query-dependent scales. Min-max normalising them into a weighted sum invents a
comparability that does not exist: the same BM25 score means something
different for a one-word query than a ten-word one, and the normalisation ends
up dominated by whichever branch returned an outlier. RRF reads only positions —
each branch contributes `1/(k + rank)` — so no score scale enters the result.

**Alternatives.** Weighted score fusion after normalisation (needs tuning per
corpus and is the failure mode above), learning-to-rank (needs training data
this project does not have), CombSUM/CombMNZ (same normalisation problem).

**Trade-off.** RRF throws away magnitude. A branch that is *overwhelmingly*
confident in its top hit gets the same `1/(k+1)` as one that is barely
confident. `RAG_RRF_K` tunes how sharply the head is favoured.

*Q: "Why k=60?"*
A: It is the value from the original Cormack et al. paper and the common
default. Larger k flattens each list's head so agreement between branches
matters more than one branch's single best hit. I did not tune it, because on a
corpus where hybrid is metric-neutral, tuning k would be fitting noise.

*Q: "Give me a property of RRF that surprised you."*
A: On fully reversed lists, a first-and-third placing slightly beats two
seconds, because `1/x` is convex: `1/61 + 1/63 > 2/62`. I asserted the opposite
in a test, the test failed, and I corrected the test rather than the code —
`test_rrf_orders_by_summed_reciprocal_rank` now documents it.

### CrossEncoder reranking

**Problem.** First-stage retrieval ranks by vector proximity, which is a
similarity signal, not a relevance judgment.

**Why.** A cross-encoder reads the query and the chunk together with full
attention, so it judges relevance rather than proximity. Measured here: **+0.090
MRR and +0.100 Recall@5**, and it fixes all three `cross_lingual` queries that
both first-stage retrievers miss entirely (Recall@5 0.000 → 1.000).

**Cost.** ~95 ms per query, and it scales with the candidate count, which is why
`RAG_RERANK_CANDIDATE_K` caps what reaches it.

**Trade-off.** Latency for quality: p50 goes from ~16 ms to ~112 ms. For an
interactive RAG answer that is dwarfed by generation; for autocomplete it would
be unacceptable.

*Q: "Where does the latency actually go?"*
A: Retrieval p50 ≈ 16 ms dense, ≈ 19 ms hybrid, ≈ 112 ms with reranking. LLM
generation on a local 4B is seconds. So the reranker is roughly 2 % of the
user-visible time and buys the largest single quality improvement in the
pipeline — an easy trade here, and the numbers are in the README.

---

## Evaluation

### Recall@k, MRR, HitRate@k

**Recall@k** — share of the relevant documents that appear in the top k. With
one relevant document it collapses to a hit rate, which is why the harness also
reports **HitRate@k** separately and carries multi-document queries.

**MRR** — mean of `1 / rank of the first relevant document`. Rewards putting the
right answer first, which is what matters when the top-k is fed to an LLM with a
context budget.

**Why both.** Recall@5 answers "is the answer reachable"; MRR answers "is it at
the top". Hybrid here changed MRR without changing Recall@5 — the two metrics
disagreeing is the finding.

*Q: "Why judge relevance at document level rather than chunk level?"*
A: Chunk ids depend on the chunking parameters. Changing `CHUNK_SIZE_WORDS`
would invalidate a chunk-level golden set, so labels are attached to documents
and survive a chunking change.

*Q: "Why no answer-quality metric?"*
A: The dataset has no reference answers. Deriving faithfulness or correctness
from document labels would be inventing a number the data cannot support. It is
listed as an absent capability, not quietly approximated.

**Limitations I would raise before the interviewer does.** The corpus is
synthetic and I wrote it alongside the system, which risks a favourable bias. 30
queries is small enough that one query moves MRR by ~0.03. The numbers are a
regression guard, not a benchmark.

---

## Agent

### LangGraph

**Problem.** Route a request between direct answering, retrieval and tools with
control flow that is inspectable rather than implicit in prompt text.

**Why.** An explicit `StateGraph` with typed state makes the control flow a data
structure — `test_graph_has_no_cycles` asserts on the compiled edges. The state
is a plain `TypedDict` of JSON types, so it serialises and is assertable in
tests.

**Deliberate non-feature: the graph is a fixed DAG, not an autonomous loop.**
`START → classify → {direct_answer | rag_search | tool_execution | fallback} →
compose_answer → END`. No edge returns to the router. `AGENT_MAX_STEPS` is a
hard ceiling every node checks anyway.

**Alternatives.** A hand-rolled if/else router (fine at this size, worse as
branches grow), LangChain agents (more magic, less inspectable), CrewAI
(multi-agent machinery this project has no use for).

*Q: "Is this a ReAct agent?"*
A: No. There is routing and tool calling, but no reason-act-observe loop and no
re-planning. I could have added a loop for the keyword; I did not, because the
honest description of a fixed DAG is more defensible than a fake loop.

*Q: "Why not multi-agent?"*
A: There is no task here that decomposes across agents. Adding a second agent
would be keyword-driven architecture.

### Tool calling via structured output

**Problem.** The local Gemma build has neither a JSON mode nor native function
calling.

**How.** A strict system prompt plus the model's JSON Schema, then the first
balanced `{...}` is located in the reply and validated with Pydantic. An invalid
reply gets **one** repair round trip that shows the model its own output and the
validation error. A second failure gives up. Two LLM calls is the hard ceiling —
no repair loop.

**Security.** The registry is an explicit allowlist of instances. No dynamic
import, no lookup by string. Arguments are validated against each tool's
Pydantic schema before the tool body runs. The calculator parses with `ast` and
walks an allowlist of node types — no `eval`, no names, no calls — with bounds
on expression length, AST depth, result magnitude and exponent. `10 ** 100000000`
is rejected on its estimated magnitude *before* being computed.

*Q: "What stops the model calling an arbitrary tool?"*
A: `ToolRegistry.get()` raises on anything not in the mapping, and the failure
comes back to the model as a controlled error. There is a test that asks for
`run_shell` and asserts it is refused.

### Persistent memory

Conversations and messages live in PostgreSQL, so history survives a restart.
Only the final assistant text is stored — the agent's internal state, routing
rationale and tool payloads never become `Message` rows, and the API response
carries no chain-of-thought. History replayed to the model is capped at
`CHAT_HISTORY_LIMIT`.

---

## Infrastructure

### Async FastAPI

Everything on the request path is I/O bound: Postgres, Qdrant, Redis, the
inference HTTP call. Async lets one process hold many in-flight requests without
a thread per request. CPU-bound work — embedding, reranking, PDF parsing — is
pushed to threads with `asyncio.to_thread` so it does not block the loop.

*Q: "Where could you still block the loop?"*
A: BM25 scoring runs inline in the event loop. On the current corpus it is
sub-millisecond; on a large one it would need `to_thread` like the models
already use.

### Celery + RabbitMQ

**Problem.** Embedding a large PDF takes minutes; an HTTP request must not wait.

**Design.** `POST /documents` validates, stores the file, writes a `processing`
row and returns **202 in ~50 ms**. Only a `document_id` travels over AMQP —
never the PDF.

**Idempotency**, which matters because `task_acks_late` redelivers from a
crashed worker: Qdrant point ids are deterministic `uuid5(namespace,
"<doc>:<index>")`, and every run clears the document's vectors and chunk rows
before writing. Verified on a live stack: a re-run produced the same two point
ids and the same two chunk rows.

**Retries** distinguish transient (Qdrant down, dropped connection → exponential
backoff with jitter) from permanent (corrupt PDF → mark failed, do not retry).

*Q: "What if a document is deleted mid-processing?"*
A: The task is revoked, but correctness does not rely on that — every heavy
stage re-checks that the row still exists and the final persist refuses to write
for a missing row. Tested on a live stack: the worker had already parsed 120
pages and embedded 360 chunks when the delete landed; it detected the missing
row, rolled back its vectors and returned `status: deleted`.

### Redis

Cache/health today, and the rate limiter's store. The limiter is a fixed window:
`INCR` plus a first-hit `EXPIRE`, both inside one Lua script so a counter can
never be left without a TTL. Keys hold a number and nothing else — authenticated
callers keyed by user id, anonymous ones by a SHA-256 prefix of their address.

*Q: "Fixed window has a boundary problem."*
A: Yes — a caller can send two full budgets across a boundary. It blunts brute
force, which is what it is for. A sliding window or token bucket would be the
upgrade.

*Q: "What if Redis is down?"*
A: It fails open and allows the request. A deliberate choice for a self-hosted
single-user deployment, documented as a limitation; a shared deployment should
flip it.

---

## Security and multi-tenancy

### JWT + Argon2id

Argon2id via `argon2-cffi` — memory-hard, per-hash salt and parameters embedded
in the digest so cost can be raised without invalidating existing hashes. Tokens
are HS256 via PyJWT with the algorithm pinned at verification, so an `alg: none`
token cannot get through (there is a test).

The payload is thin on purpose: `sub`, `role`, `iat`, `exp`, `iss`. No email, no
name — a leaked token should reveal as little as possible.

Login does not distinguish "no such account" from "wrong password": the unknown
path still hashes a dummy digest so timing does not turn the endpoint into an
account oracle.

*Q: "No refresh tokens?"*
A: Correct, and listed as a limitation. Deactivating a user takes effect
immediately because every request re-reads the row, but an individual stolen
token cannot be revoked before it expires.

### Qdrant tenant filtering

**The important one.** Every point carries `user_id` in its payload with a
payload index, and `VectorStore.search()` takes `user_id` as a **required**
argument — it cannot be called unscoped. `upsert_chunks()` refuses a record
without one.

**Filtering happens inside Qdrant, not in Python afterwards.** That is not a
performance detail: with a post-filter, a query whose nearest neighbour belongs
to another tenant comes back short — or empty at `limit=1` — and the foreign
chunk has already been read.

`tests/test_tenant_integration.py` proves it against a live Qdrant: two tenants
at opposite corners of the vector space, a query vector nearest to tenant B, and
`limit=1` as tenant A. A post-filter would return nothing; server-side filtering
returns A's own best chunk.

**Ownership is enforced at the schema level too** — `user_id` is NOT NULL on
both `conversations` and `documents`. A foreign resource returns **404, not
403**, so the API never confirms that someone else's UUID exists.

*Q: "Admin bypasses this, right?"*
A: No. Admin is a separate `/admin` surface; an admin calling `GET /documents`
still sees only their own. A bug in a user route therefore cannot quietly become
a tenant bypass.

---

## Local inference

### MLX / Gemma-3-4B, 4-bit QAT

**Why local.** The entire premise: documents, embeddings, conversations and
generation stay on the machine.

**Why MLX.** Apple Silicon unified memory; MLX is the native runtime. The model
is `mlx-community/gemma-3-4b-it-qat-4bit`.

**Be precise about quantization.** I *use* a 4-bit quantization-aware-trained
model. I did **not** quantize it. QAT means quantization was simulated during
training, so it degrades less than post-training quantization at the same bit
width. 4-bit gets a 4B model into a laptop's memory budget alongside the
embedding and reranker models; the cost is some quality against fp16.

**Trade-offs.** One process, generation serialised behind a lock, so concurrent
chats queue. No continuous batching, no paged attention — which is exactly what
vLLM or SGLang provide, and why I would reach for them on a GPU server.

*Q: "Why not vLLM?"*
A: vLLM targets CUDA. This runs on Apple Silicon, where MLX is the right
runtime. On an NVIDIA deployment vLLM would be the obvious choice for continuous
batching and paged-attention KV cache — I have not used it, and I would not
claim otherwise.

*Q: "What would you measure before choosing?"*
A: Tokens per second at realistic concurrency, time to first token, and memory
headroom against the KV cache at the target context length. The service already
reports `generation_time_seconds` and token counts, so throughput is measurable
today.

---

## Observability

### Three layers

Prometheus answers "is the service healthy" — aggregates and alerting.
Structured logs answer "what happened to request X". Langfuse answers "where did
this answer come from" — stage timings, routing decisions, retrieval counts,
token usage. A single `request_id` ties them together; the Langfuse trace id is
derived from it.

*Q: "Why not put user_id in a Prometheus label?"*
A: Cardinality. Prometheus creates one time series per label combination, so an
unbounded label grows the series count with traffic until the scrape target runs
out of memory. Identifiers belong in logs and traces, which are built to hold
them. Every label in this project comes from a closed set: one or two models, a
few statuses, four registered tools.

### Fail-open tracing

**Principle.** Tracing is diagnostic. It must never be a dependency of request
execution. Every Langfuse interaction is wrapped; an unreachable host, invalid
credentials, a missing package or an export failure is logged **once**, the
tracer marks itself degraded, and requests continue untraced.

The "once" is deliberate — a broken exporter must not turn an observability
problem into a log-volume problem. `test_a_failing_exporter_degrades_once`
asserts exactly one warning across five failing spans.

*Q: "How do you know it actually fails open?"*
A: The fake Langfuse client can be told to raise on span start. There are tests
that run full retrieval and a full LLM call through a tracer backed by it and
assert the results are unaffected.

*Q: "What is the overhead when tracing is off?"*
A: A no-op context manager per stage. The retrieval evaluation was re-run after
instrumentation and the metrics are identical — MRR 0.888/0.978/0.879/0.978 —
with latency inside run-to-run noise.

### Privacy defaults

Spans carry metadata only: request id, route, mode, model, candidate counts,
`top_k`, durations, tool names, statuses, token counts. Never JWTs, the
`Authorization` header, API keys, passwords, environment variables or raw
documents. Exception *messages* are not recorded either — only the class,
because a message can carry a path or a row of data.

`LANGFUSE_CAPTURE_CONTENT` additionally sends prompts, questions, chunk text and
answers. Off by default, truncated when on.

*Q: "Why is content capture off? It is the most useful part."*
A: It is, for debugging. It is also user content leaving the machine, which
contradicts the project's premise. Making it an explicit opt-in means the person
turning it on has decided that the target instance is acceptable for that data.

### Token accounting

Recorded **only when the inference service reports it**. It does: mlx-vlm's
`GenerationResult` exposes `prompt_tokens`, `generation_tokens` and
`total_tokens` from the tokenizer that just ran, and the service passes them
through tagged `source: "local_tokenizer"`.

That tag matters: these are **locally measured counts from the serving
process**, not a provider's billing-grade usage. If a response carries no usage,
or a partial one, the backend records none — `_usage_from()` returns `None`
rather than estimating. An estimate presented as usage makes the metric quietly
wrong, which is worse than not having it.

**No cost is calculated.** The model is local; there is no per-token price. A
fabricated cost figure would be worse than none.

*Q: "Could you not estimate tokens with a tokenizer on the client side?"*
A: I could, and it would usually be close. But it would be a *different*
tokenizer run on a *different* string than the one the model saw — no chat
template, no special tokens. Labelling that as usage would be misleading, so if
a provider reports nothing, the field stays empty.
