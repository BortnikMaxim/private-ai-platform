"""The reset_qdrant maintenance utility must never destroy data by accident."""

import argparse
from types import SimpleNamespace

import pytest

from backend.scripts import reset_qdrant as script


class FakeQdrantClient:
    def __init__(self, document_ids: list[str], exists: bool = True) -> None:
        self.points = [
            SimpleNamespace(payload={"document_id": document_id})
            for document_id in document_ids
        ]
        self.exists = exists
        self.deleted_filters: list[object] = []
        self.deleted_collections: list[str] = []
        self.created_collections: list[str] = []
        self.closed = False

    async def collection_exists(self, _name: str) -> bool:
        return self.exists

    async def get_collection(self, _name: str):
        return SimpleNamespace(points_count=len(self.points))

    async def scroll(self, collection_name, limit, offset, with_payload, with_vectors):
        return list(self.points), None

    async def delete(self, collection_name, points_selector, wait=True):
        self.deleted_filters.append(points_selector)
        matched = {
            value
            for condition in points_selector.filter.must
            for value in condition.match.any
        }
        self.points = [
            point for point in self.points if point.payload["document_id"] not in matched
        ]

    async def delete_collection(self, name: str) -> None:
        self.deleted_collections.append(name)
        self.points = []
        self.exists = False

    async def create_collection(self, collection_name, vectors_config) -> None:
        self.created_collections.append(collection_name)
        self.vectors_config = vectors_config
        self.exists = True

    async def create_payload_index(self, **_kwargs) -> None:
        return None

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_client(monkeypatch):
    client = FakeQdrantClient(document_ids=["known-1", "known-1", "orphan-9"])
    monkeypatch.setattr(script, "AsyncQdrantClient", lambda **_kwargs: client)

    async def known(_url):
        return {"known-1"}

    monkeypatch.setattr(script, "_known_document_ids", known)

    return client


def args(**overrides) -> argparse.Namespace:
    return argparse.Namespace(
        **{"purge_orphans": False, "recreate": False, "yes": False, **overrides}
    )


async def test_default_mode_only_reports(fake_client, capsys):
    exit_code = await script.run(args())

    assert exit_code == 0
    assert fake_client.deleted_filters == []
    assert fake_client.deleted_collections == []
    assert len(fake_client.points) == 3

    output = capsys.readouterr().out
    assert "orphan-9" in output
    assert "Nothing changed" in output


async def test_purge_without_confirmation_changes_nothing(fake_client):
    exit_code = await script.run(args(purge_orphans=True))

    assert exit_code == 1
    assert fake_client.deleted_filters == []
    assert len(fake_client.points) == 3


async def test_recreate_without_confirmation_changes_nothing(fake_client):
    exit_code = await script.run(args(recreate=True))

    assert exit_code == 1
    assert fake_client.deleted_collections == []
    assert len(fake_client.points) == 3


async def test_purge_with_confirmation_removes_only_orphans(fake_client):
    exit_code = await script.run(args(purge_orphans=True, yes=True))

    assert exit_code == 0
    assert [point.payload["document_id"] for point in fake_client.points] == [
        "known-1",
        "known-1",
    ]


async def test_recreate_with_confirmation_rebuilds_the_collection(fake_client, settings):
    exit_code = await script.run(args(recreate=True, yes=True))

    assert exit_code == 0
    assert fake_client.deleted_collections == [script.get_settings().qdrant_collection]
    assert fake_client.created_collections == [script.get_settings().qdrant_collection]
    # Rebuilt with the configured vector size and cosine distance.
    assert fake_client.vectors_config.size == script.get_settings().embedding_dim
    assert fake_client.vectors_config.distance == "Cosine"


async def test_it_touches_only_the_configured_collection(fake_client):
    await script.run(args(recreate=True, yes=True))

    collection = script.get_settings().qdrant_collection
    assert set(fake_client.deleted_collections) == {collection}
    assert set(fake_client.created_collections) == {collection}


async def test_purge_is_refused_when_postgres_is_unavailable(monkeypatch, fake_client):
    async def unknown(_url):
        return None

    monkeypatch.setattr(script, "_known_document_ids", unknown)

    exit_code = await script.run(args(purge_orphans=True, yes=True))

    # Without the documents table every vector would look orphaned.
    assert exit_code == 2
    assert fake_client.deleted_filters == []
    assert len(fake_client.points) == 3


async def test_unreachable_qdrant_reports_instead_of_raising(monkeypatch):
    class Unreachable(FakeQdrantClient):
        async def collection_exists(self, _name):
            raise ConnectionError("connection refused")

    monkeypatch.setattr(script, "AsyncQdrantClient", lambda **_kwargs: Unreachable([]))

    assert await script.run(args()) == 2


def test_conflicting_flags_are_rejected(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["reset_qdrant", "--purge-orphans", "--recreate", "--yes"],
    )

    with pytest.raises(SystemExit) as error:
        script.main()

    assert error.value.code == 2
