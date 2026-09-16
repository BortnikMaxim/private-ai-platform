"""Tenant isolation against a real Qdrant.

The unit tests prove the code passes a ``user_id`` down; this proves Qdrant
actually enforces it. The decisive case is the one where user B's chunk is the
*nearest* neighbour of user A's query: a Python filter applied after a limited
search would silently return fewer results (or, if the limit were 1, nothing),
whereas a server-side filter returns A's own best chunk. Only the latter is
both correct and safe.

    docker compose up -d
    pytest -m integration -o addopts=""
"""

import uuid

import pytest
from qdrant_client import AsyncQdrantClient

from backend.config import get_settings
from backend.services.vector_store import VectorStore

pytestmark = pytest.mark.integration

# Four dimensions is enough to place two tenants at opposite corners.
ALICE_VECTORS = [
    [1.0, 0.0, 0.0, 0.0],
    [0.9, 0.1, 0.0, 0.0],
    [0.8, 0.2, 0.0, 0.0],
]
BOB_VECTORS = [
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.9, 0.1, 0.0],
    [0.0, 0.8, 0.2, 0.0],
]


def _chunks(user_id: str, document_id: str, filename: str, vectors) -> list[dict]:
    return [
        {
            "point_id": str(uuid.uuid4()),
            "vector": vector,
            "user_id": user_id,
            "document_id": document_id,
            "filename": filename,
            "page": 1,
            "chunk_index": index,
            "text": f"{filename} secret chunk {index}",
        }
        for index, vector in enumerate(vectors)
    ]


@pytest.fixture
async def store():
    settings = get_settings()
    collection = f"tenant_itest_{uuid.uuid4().hex[:8]}"

    client = AsyncQdrantClient(url=settings.qdrant_url)
    vector_store = VectorStore(
        client=client,
        collection_name=collection,
        vector_size=4,
    )

    try:
        await vector_store.ensure_collection()
    except Exception as exc:  # noqa: BLE001
        await client.close()
        pytest.skip(f"Qdrant is not reachable: {exc}")

    try:
        yield vector_store
    finally:
        await client.delete_collection(collection)
        await client.close()


@pytest.fixture
async def two_tenants(store):
    alice, bob = str(uuid.uuid4()), str(uuid.uuid4())
    alice_doc, bob_doc = str(uuid.uuid4()), str(uuid.uuid4())

    await store.upsert_chunks(_chunks(alice, alice_doc, "alice.pdf", ALICE_VECTORS))
    await store.upsert_chunks(_chunks(bob, bob_doc, "bob.pdf", BOB_VECTORS))

    return {
        "alice": alice,
        "bob": bob,
        "alice_doc": alice_doc,
        "bob_doc": bob_doc,
    }


async def test_retrieval_returns_only_the_querying_tenant(store, two_tenants):
    alice_hits = await store.search(
        vector=ALICE_VECTORS[0], limit=10, user_id=two_tenants["alice"]
    )
    bob_hits = await store.search(
        vector=BOB_VECTORS[0], limit=10, user_id=two_tenants["bob"]
    )

    assert len(alice_hits) == 3
    assert {hit["user_id"] for hit in alice_hits} == {two_tenants["alice"]}
    assert {hit["filename"] for hit in alice_hits} == {"alice.pdf"}

    assert len(bob_hits) == 3
    assert {hit["user_id"] for hit in bob_hits} == {two_tenants["bob"]}
    assert {hit["filename"] for hit in bob_hits} == {"bob.pdf"}


async def test_a_closer_foreign_chunk_is_still_never_returned(store, two_tenants):
    """Alice queries with a vector that is nearest to Bob's data."""
    unfiltered = await store.search(
        vector=BOB_VECTORS[0], limit=10, user_id=two_tenants["bob"]
    )
    assert unfiltered[0]["filename"] == "bob.pdf", "bob should dominate this query"

    # Same query vector, but as Alice.
    hits = await store.search(
        vector=BOB_VECTORS[0], limit=10, user_id=two_tenants["alice"]
    )

    assert len(hits) == 3
    assert {hit["filename"] for hit in hits} == {"alice.pdf"}
    assert not any("bob" in (hit["text"] or "") for hit in hits)


async def test_the_filter_runs_inside_qdrant_not_afterwards(store, two_tenants):
    """limit=1 on a query whose nearest neighbour belongs to the other tenant.

    Post-filtering would fetch Bob's chunk and then discard it, leaving an empty
    result. Getting Alice's own best chunk back proves Qdrant filtered while
    searching.
    """
    hits = await store.search(
        vector=BOB_VECTORS[0], limit=1, user_id=two_tenants["alice"]
    )

    assert len(hits) == 1
    assert hits[0]["user_id"] == two_tenants["alice"]
    assert hits[0]["filename"] == "alice.pdf"


async def test_naming_a_foreign_document_returns_nothing(store, two_tenants):
    """The document scope narrows; it can never widen past the tenant."""
    hits = await store.search(
        vector=BOB_VECTORS[0],
        limit=10,
        user_id=two_tenants["alice"],
        document_ids=[two_tenants["bob_doc"]],
    )

    assert hits == []


async def test_both_filters_combine(store, two_tenants):
    hits = await store.search(
        vector=ALICE_VECTORS[0],
        limit=10,
        user_id=two_tenants["alice"],
        document_ids=[two_tenants["alice_doc"]],
    )

    assert len(hits) == 3
    assert {hit["document_id"] for hit in hits} == {two_tenants["alice_doc"]}


async def test_deleting_one_tenants_document_leaves_the_other_intact(
    store,
    two_tenants,
):
    await store.delete_document(two_tenants["bob_doc"], user_id=two_tenants["bob"])

    alice_hits = await store.search(
        vector=ALICE_VECTORS[0], limit=10, user_id=two_tenants["alice"]
    )
    bob_hits = await store.search(
        vector=BOB_VECTORS[0], limit=10, user_id=two_tenants["bob"]
    )

    assert len(alice_hits) == 3
    assert bob_hits == []


async def test_a_delete_scoped_to_the_wrong_tenant_removes_nothing(store, two_tenants):
    await store.delete_document(two_tenants["bob_doc"], user_id=two_tenants["alice"])

    bob_hits = await store.search(
        vector=BOB_VECTORS[0], limit=10, user_id=two_tenants["bob"]
    )

    assert len(bob_hits) == 3


async def test_untagged_points_are_refused_at_write_time(store):
    with pytest.raises(ValueError, match="user_id"):
        await store.upsert_chunks(
            [
                {
                    "point_id": str(uuid.uuid4()),
                    "vector": [1.0, 0.0, 0.0, 0.0],
                    "document_id": str(uuid.uuid4()),
                    "filename": "orphan.pdf",
                    "page": 1,
                    "chunk_index": 0,
                    "text": "no owner",
                }
            ]
        )


async def test_the_payload_index_covers_the_tenant_field(store):
    info = await store.client.get_collection(store.collection_name)

    assert "user_id" in info.payload_schema
    assert "document_id" in info.payload_schema
