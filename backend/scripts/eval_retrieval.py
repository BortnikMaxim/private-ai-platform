"""Offline retrieval evaluation.

Indexes a fixed corpus into a throwaway Qdrant collection and a throwaway
PostgreSQL tenant, runs the golden query set through several retrieval
configurations, and reports Recall@k, MRR, HitRate@k and retrieval latency.

    python -m backend.scripts.eval_retrieval

Only retrieval is measured. No LLM is called anywhere in this script, so the
latency numbers contain no generation time.

Relevance is judged at document level: a query counts as answered when a
retrieved chunk belongs to one of its ``expected_document_ids``. The dataset
does not carry reference answers, so no answer-quality metric is computed —
inventing one from these labels would be dishonest.

Everything is torn down afterwards: the scratch collection is deleted and the
scratch rows are removed.
"""

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
import uuid
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qdrant_client import AsyncQdrantClient
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker

from backend.config import Settings, get_settings
from backend.db import create_engine
from backend.models import Document, DocumentChunk, User
from backend.services.chunking import build_chunks
from backend.services.document_processor import point_id_for
from backend.services.embeddings import EmbeddingService
from backend.services.lexical_index import LexicalRetriever
from backend.services.rag_service import MODE_DENSE, MODE_HYBRID, RagService
from backend.services.vector_store import VectorStore

DATASET_DIR = Path(__file__).resolve().parents[2] / "eval" / "dataset"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "eval" / "results"

K_VALUES = (1, 3, 5, 10)

# Queries run before timing starts, to keep start-up cost out of the latency.
WARMUP_QUERIES = 3


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Share of the relevant documents that appear in the top k."""
    if not relevant:
        return 0.0

    found = {doc for doc in retrieved[:k] if doc in relevant}

    return len(found) / len(relevant)


def hit_rate_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """1.0 if at least one relevant document is in the top k."""
    return 1.0 if any(doc in relevant for doc in retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant document, 0 if it never appears."""
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / rank

    return 0.0


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile: the ceil(fraction * N)-th smallest sample.

    Nearest rank rather than linear interpolation, because interpolating
    between two of thirty samples invents a latency that was never measured.
    """
    if not values:
        return 0.0

    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))

    return ordered[index]


# ---------------------------------------------------------------------------
# Configurations under test
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunConfig:
    name: str
    mode: str
    rerank: bool
    description: str


CONFIGS = (
    RunConfig(
        name="dense",
        mode=MODE_DENSE,
        rerank=False,
        description="multilingual-e5-small + Qdrant, no reranking",
    ),
    RunConfig(
        name="dense+rerank",
        mode=MODE_DENSE,
        rerank=True,
        description="dense candidates reranked by the cross-encoder",
    ),
    RunConfig(
        name="hybrid",
        mode=MODE_HYBRID,
        rerank=False,
        description="dense + BM25 fused with RRF, no reranking",
    ),
    RunConfig(
        name="hybrid+rerank",
        mode=MODE_HYBRID,
        rerank=True,
        description="dense + BM25 fused with RRF, then cross-encoder reranking",
    ),
)


@dataclass
class QueryOutcome:
    query_id: str
    category: str
    retrieved: list[str]
    reciprocal_rank: float
    latency_ms: float
    recall: dict[int, float] = field(default_factory=dict)
    hit_rate: dict[int, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Fixture lifecycle
# ---------------------------------------------------------------------------


async def index_corpus(
    session_factory: async_sessionmaker,
    vector_store: VectorStore,
    embeddings: EmbeddingService,
    settings: Settings,
    documents: list[dict[str, Any]],
    owner_id: uuid.UUID,
) -> int:
    """Chunk, embed and index the corpus exactly as the worker would."""
    total_chunks = 0

    for entry in documents:
        document_id = uuid.uuid5(uuid.NAMESPACE_URL, f"eval/{entry['id']}")
        pages = [{"page": 1, "text": entry["text"]}]
        records = build_chunks(
            pages,
            chunk_size=settings.chunk_size_words,
            overlap=settings.chunk_overlap_words,
        )

        vectors = await embeddings.embed_passages([r["text"] for r in records])

        points = [
            {
                "point_id": point_id_for(document_id, record["chunk_index"]),
                "vector": vector,
                "user_id": str(owner_id),
                "document_id": str(document_id),
                "filename": entry["filename"],
                "page": record["page"],
                "chunk_index": record["chunk_index"],
                "text": record["text"],
            }
            for record, vector in zip(records, vectors, strict=True)
        ]

        await vector_store.upsert_chunks(points)

        async with session_factory() as session:
            session.add(
                Document(
                    id=document_id,
                    user_id=owner_id,
                    filename=entry["filename"],
                    original_filename=entry["filename"],
                    content_type="application/pdf",
                    size_bytes=len(entry["text"].encode("utf-8")),
                    status="ready",
                    total_pages=1,
                    extracted_pages=1,
                    chunks_count=len(points),
                )
            )
            session.add_all(
                [
                    DocumentChunk(
                        document_id=document_id,
                        qdrant_point_id=point["point_id"],
                        page=point["page"],
                        chunk_index=point["chunk_index"],
                        text=point["text"],
                    )
                    for point in points
                ]
            )
            await session.commit()

        total_chunks += len(points)

    return total_chunks


async def cleanup(
    session_factory: async_sessionmaker,
    client: AsyncQdrantClient,
    collection: str,
    owner_id: uuid.UUID,
) -> None:
    # Teardown must never mask the result of the run itself.
    with suppress(Exception):
        async with session_factory() as session:
            # documents cascade to their chunks and the user cascades to the
            # documents, so one delete removes the whole scratch tenant.
            await session.execute(delete(User).where(User.id == owner_id))
            await session.commit()

    with suppress(Exception):
        await client.delete_collection(collection)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


async def evaluate_config(
    rag: RagService,
    session_factory: async_sessionmaker,
    config: RunConfig,
    queries: list[dict[str, Any]],
    owner_id: uuid.UUID,
    top_k: int,
) -> tuple[list[QueryOutcome], dict[str, Any]]:
    outcomes: list[QueryOutcome] = []

    # Warm-up, excluded from the measurements. The first call through a
    # configuration pays for lazy tokenizer setup, the first ONNX/torch graph
    # execution and the BM25 index build; folding that into p95 would report
    # a start-up cost as if it were steady-state latency.
    async with session_factory() as session:
        for entry in queries[:WARMUP_QUERIES]:
            await rag.retrieve(
                question=entry["query"],
                user_id=str(owner_id),
                session=session,
                mode=config.mode,
                rerank=config.rerank,
                top_k=top_k,
            )

    for entry in queries:
        relevant = {
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"eval/{doc}"))
            for doc in entry["expected_document_ids"]
        }

        async with session_factory() as session:
            started = time.perf_counter()
            hits = await rag.retrieve(
                question=entry["query"],
                user_id=str(owner_id),
                session=session,
                mode=config.mode,
                rerank=config.rerank,
                top_k=top_k,
            )
            latency_ms = (time.perf_counter() - started) * 1000

        # De-duplicate to document level, preserving rank order.
        retrieved: list[str] = []
        for hit in hits:
            document_id = str(hit.get("document_id"))
            if document_id not in retrieved:
                retrieved.append(document_id)

        outcome = QueryOutcome(
            query_id=entry["id"],
            category=entry.get("category", "uncategorised"),
            retrieved=retrieved,
            reciprocal_rank=reciprocal_rank(retrieved, relevant),
            latency_ms=latency_ms,
        )

        for k in K_VALUES:
            outcome.recall[k] = recall_at_k(retrieved, relevant, k)
            outcome.hit_rate[k] = hit_rate_at_k(retrieved, relevant, k)

        outcomes.append(outcome)

    latencies = [outcome.latency_ms for outcome in outcomes]

    summary: dict[str, Any] = {
        "config": config.name,
        "mode": config.mode,
        "rerank": config.rerank,
        "description": config.description,
        "queries": len(outcomes),
        "mrr": round(statistics.fmean(o.reciprocal_rank for o in outcomes), 4),
        "recall": {
            f"@{k}": round(statistics.fmean(o.recall[k] for o in outcomes), 4)
            for k in K_VALUES
        },
        "hit_rate": {
            f"@{k}": round(statistics.fmean(o.hit_rate[k] for o in outcomes), 4)
            for k in K_VALUES
        },
        "latency_ms": {
            "p50": round(percentile(latencies, 0.50), 1),
            "p95": round(percentile(latencies, 0.95), 1),
            "mean": round(statistics.fmean(latencies), 1),
        },
    }

    by_category: dict[str, list[QueryOutcome]] = {}
    for outcome in outcomes:
        by_category.setdefault(outcome.category, []).append(outcome)

    summary["by_category"] = {
        category: {
            "queries": len(group),
            "mrr": round(statistics.fmean(o.reciprocal_rank for o in group), 4),
            "recall@5": round(statistics.fmean(o.recall[5] for o in group), 4),
        }
        for category, group in sorted(by_category.items())
    }

    return outcomes, summary


def print_report(summaries: list[dict[str, Any]], baseline: str = "dense") -> None:
    print("\n" + "=" * 96)
    print("RETRIEVAL EVALUATION")
    print("=" * 96)

    header = (
        f"{'config':<16}{'MRR':>8}{'R@1':>8}{'R@3':>8}"
        f"{'R@5':>8}{'H@5':>8}{'p50 ms':>10}{'p95 ms':>10}"
    )
    print(header)
    print("-" * 96)

    for summary in summaries:
        print(
            f"{summary['config']:<16}"
            f"{summary['mrr']:>8.3f}"
            f"{summary['recall']['@1']:>8.3f}"
            f"{summary['recall']['@3']:>8.3f}"
            f"{summary['recall']['@5']:>8.3f}"
            f"{summary['hit_rate']['@5']:>8.3f}"
            f"{summary['latency_ms']['p50']:>10.1f}"
            f"{summary['latency_ms']['p95']:>10.1f}"
        )

    reference = next((s for s in summaries if s["config"] == baseline), None)

    if reference is not None:
        print("\nChange vs. the dense baseline (MRR / Recall@5):")
        for summary in summaries:
            if summary["config"] == baseline:
                continue
            mrr_delta = summary["mrr"] - reference["mrr"]
            recall_delta = summary["recall"]["@5"] - reference["recall"]["@5"]
            print(
                f"  {summary['config']:<16}"
                f"MRR {mrr_delta:+.3f}   Recall@5 {recall_delta:+.3f}"
            )

    print("\nRecall@5 by query category:")
    categories = sorted({c for s in summaries for c in s["by_category"]})
    print(f"  {'category':<16}" + "".join(f"{s['config']:>16}" for s in summaries))

    for category in categories:
        row = f"  {category:<16}"
        for summary in summaries:
            value = summary["by_category"].get(category)
            row += f"{value['recall@5']:>16.3f}" if value else f"{'-':>16}"
        print(row)

    print("=" * 96 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()

    corpus = json.loads((DATASET_DIR / "corpus.json").read_text(encoding="utf-8"))
    golden = json.loads((DATASET_DIR / "queries.json").read_text(encoding="utf-8"))
    documents = corpus["documents"]
    queries = golden["queries"]

    collection = f"eval_{uuid.uuid4().hex[:8]}"
    owner_id = uuid.uuid4()

    engine = create_engine(settings.database_url)
    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    client = AsyncQdrantClient(url=settings.qdrant_url)

    vector_store = VectorStore(
        client=client,
        collection_name=collection,
        vector_size=settings.embedding_dim,
    )
    embeddings = EmbeddingService(
        embedding_model=settings.embedding_model,
        reranker_model=settings.reranker_model,
    )

    try:
        await vector_store.ensure_collection()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Qdrant is not reachable at {settings.qdrant_url}: {exc}")
        await client.close()
        await engine.dispose()
        return 2

    print(f"Loading models ({settings.embedding_model} + {settings.reranker_model})...")
    await embeddings.ensure_loaded()

    try:
        async with session_factory() as session:
            session.add(
                User(
                    id=owner_id,
                    email=f"eval-{owner_id.hex[:8]}@local.invalid",
                    password_hash="!eval-no-login",
                    is_active=False,
                )
            )
            await session.commit()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: PostgreSQL is not reachable or not migrated: {exc}")
        await client.close()
        await engine.dispose()
        return 2

    try:
        print(f"Indexing {len(documents)} documents into '{collection}'...")
        chunk_count = await index_corpus(
            session_factory, vector_store, embeddings, settings, documents, owner_id
        )
        print(f"Indexed {chunk_count} chunks.")

        rag = RagService(
            embeddings=embeddings,
            vector_store=vector_store,
            settings=settings,
            lexical=LexicalRetriever(
                k1=settings.bm25_k1,
                b=settings.bm25_b,
                stemming=settings.bm25_stemming,
            ),
        )

        summaries: list[dict[str, Any]] = []
        details: dict[str, Any] = {}

        for config in CONFIGS:
            print(f"Running '{config.name}' over {len(queries)} queries...")
            outcomes, summary = await evaluate_config(
                rag, session_factory, config, queries, owner_id, args.top_k
            )
            summaries.append(summary)
            details[config.name] = [asdict(outcome) for outcome in outcomes]

        print_report(summaries)

        payload = {
            "generated_at": datetime.now(UTC).isoformat(),
            "dataset": {
                "documents": len(documents),
                "chunks": chunk_count,
                "queries": len(queries),
                "corpus_file": "eval/dataset/corpus.json",
                "queries_file": "eval/dataset/queries.json",
            },
            "configuration": {
                "embedding_model": settings.embedding_model,
                "reranker_model": settings.reranker_model,
                "embedding_dim": settings.embedding_dim,
                "chunk_size_words": settings.chunk_size_words,
                "chunk_overlap_words": settings.chunk_overlap_words,
                "rag_candidate_k": settings.rag_candidate_k,
                "rag_lexical_candidate_k": settings.rag_lexical_candidate_k,
                "rag_rrf_k": settings.rag_rrf_k,
                "rag_rerank_candidate_k": settings.rag_rerank_candidate_k,
                "bm25_k1": settings.bm25_k1,
                "bm25_b": settings.bm25_b,
                "bm25_stemming": settings.bm25_stemming,
                "top_k": args.top_k,
            },
            "note": (
                "Retrieval only. No LLM call is made, so latency excludes "
                "generation. The first "
                f"{WARMUP_QUERIES} queries of each configuration are run as a "
                "warm-up and excluded from the timings. Relevance is judged at "
                "document level; the dataset carries no reference answers, so "
                "no answer-quality metric is reported."
            ),
            "warmup_queries": WARMUP_QUERIES,
            "results": summaries,
        }

        if args.details:
            payload["per_query"] = details

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output = RESULTS_DIR / args.output
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Machine-readable results written to {output}")

        return 0

    finally:
        await cleanup(session_factory, client, collection, owner_id)
        await client.close()
        await engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backend.scripts.eval_retrieval",
        description="Offline retrieval evaluation over the golden dataset.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="documents scored per query")
    parser.add_argument("--output", default="latest.json", help="file under eval/results/")
    parser.add_argument(
        "--details",
        action="store_true",
        help="include per-query outcomes in the JSON output",
    )

    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
