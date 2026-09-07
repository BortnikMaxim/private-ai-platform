"""Opt-in tests against the real Docker services.

    docker compose up -d
    alembic upgrade head
    pytest -m integration

They are excluded from the default `pytest -q` run because they need live
PostgreSQL and Qdrant.
"""

import uuid

import pytest
from qdrant_client import AsyncQdrantClient
from sqlalchemy import inspect, text

from backend.config import get_settings
from backend.db import create_engine
from backend.services.vector_store import VectorStore

pytestmark = pytest.mark.integration

EXPECTED_TABLES = {
    "users",
    "conversations",
    "messages",
    "documents",
    "document_chunks",
}


async def test_postgres_schema_matches_the_models():
    engine = create_engine()

    try:
        async with engine.connect() as connection:
            assert (await connection.execute(text("SELECT 1"))).scalar_one() == 1

            tables = set(
                await connection.run_sync(
                    lambda sync_connection: inspect(sync_connection).get_table_names()
                )
            )
    except OSError as exc:
        pytest.skip(f"PostgreSQL is not reachable: {exc}")
    finally:
        await engine.dispose()

    missing = EXPECTED_TABLES - tables

    if missing:
        pytest.fail(f"Missing tables {sorted(missing)}; run `alembic upgrade head`")


def _chunks(document_id: str, filename: str, vectors: list[list[float]]) -> list[dict]:
    return [
        {
            "point_id": str(uuid.uuid4()),
            "vector": vector,
            "document_id": document_id,
            "filename": filename,
            "page": 1,
            "chunk_index": index,
            "text": f"{filename} chunk {index}",
        }
        for index, vector in enumerate(vectors)
    ]


async def test_qdrant_filtering_is_server_side_and_never_leaks_other_documents():
    """Two documents, retrieval scoped to one, nothing from the other comes back.

    The decisive assertion is the `limit=1` case: the query vector is closest to
    doc2, so a Python filter applied *after* a limit-1 retrieval would return
    nothing. Getting a doc1 chunk back proves Qdrant applied the filter while
    searching.
    """
    settings = get_settings()
    collection = f"itest_{uuid.uuid4().hex[:8]}"

    client = AsyncQdrantClient(url=settings.qdrant_url)
    store = VectorStore(client=client, collection_name=collection, vector_size=4)

    try:
        await store.ensure_collection()
    except Exception as exc:  # noqa: BLE001
        await client.close()
        pytest.skip(f"Qdrant is not reachable: {exc}")

    doc1 = str(uuid.uuid4())
    doc2 = str(uuid.uuid4())

    try:
        await store.upsert_chunks(
            _chunks(doc1, "doc1.pdf", [[1.0, 0.0, 0.0, 0.0],
                                       [0.9, 0.1, 0.0, 0.0],
                                       [0.8, 0.2, 0.0, 0.0]])
        )
        await store.upsert_chunks(
            _chunks(doc2, "doc2.pdf", [[0.0, 1.0, 0.0, 0.0],
                                       [0.0, 0.9, 0.1, 0.0],
                                       [0.0, 0.8, 0.2, 0.0]])
        )

        query = [0.0, 1.0, 0.0, 0.0]  # nearest neighbours all belong to doc2

        unfiltered = await store.search(query, limit=10)
        assert len(unfiltered) == 6
        assert unfiltered[0]["document_id"] == doc2, "doc2 should dominate unfiltered"

        scoped = await store.search(query, limit=10, document_ids=[doc1])

        assert len(scoped) == 3
        assert {hit["document_id"] for hit in scoped} == {doc1}
        assert not any(hit["document_id"] == doc2 for hit in scoped)
        assert {hit["filename"] for hit in scoped} == {"doc1.pdf"}
        assert sorted(hit["chunk_index"] for hit in scoped) == [0, 1, 2]

        # Server-side proof: only Qdrant-side filtering can return a doc1 hit
        # when the single nearest neighbour belongs to doc2.
        narrow = await store.search(query, limit=1, document_ids=[doc1])
        assert len(narrow) == 1
        assert narrow[0]["document_id"] == doc1

        # Filtering on both documents restores the full result set.
        both = await store.search(query, limit=10, document_ids=[doc1, doc2])
        assert len(both) == 6

        # Filtering on an unknown document yields nothing rather than everything.
        unknown = await store.search(query, limit=10, document_ids=[str(uuid.uuid4())])
        assert unknown == []

        # Deleting doc2 leaves doc1 untouched.
        await store.delete_document(doc2)

        remaining = await store.search(query, limit=10)
        assert len(remaining) == 3
        assert {hit["document_id"] for hit in remaining} == {doc1}

    finally:
        await client.delete_collection(collection)
        await client.close()
