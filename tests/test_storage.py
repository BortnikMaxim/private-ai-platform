"""DocumentStorage: path safety and file lifecycle."""

import uuid

import pytest

from backend.errors import DocumentSourceMissingError
from backend.services.storage import DocumentStorage


@pytest.fixture
def store(tmp_path) -> DocumentStorage:
    storage = DocumentStorage(tmp_path / "uploads")
    storage.ensure_ready()
    return storage


def test_path_is_derived_from_the_uuid_only(store):
    document_id = uuid.uuid4()
    path = store.path_for(document_id)

    assert path.parent == store.base_dir
    assert path.name == f"{document_id}.pdf"


@pytest.mark.parametrize(
    "malicious",
    [
        "../../etc/passwd",
        "..",
        "/etc/passwd",
        "a/b",
        "\x00",
        "..%2f..%2fetc",
        "",
    ],
)
def test_traversal_attempts_cannot_produce_a_path(store, malicious):
    # Names are parsed as UUIDs, so nothing that is not a UUID gets through.
    with pytest.raises(ValueError):
        store.path_for(malicious)


def test_a_document_id_can_never_escape_the_base_directory(store):
    for _ in range(200):
        path = store.path_for(uuid.uuid4())
        assert store.base_dir in path.parents


async def test_save_read_delete_roundtrip(store):
    document_id = uuid.uuid4()

    assert store.exists(document_id) is False

    await store.save(document_id, b"%PDF-1.4 body")

    assert store.exists(document_id) is True
    assert await store.read(document_id) == b"%PDF-1.4 body"

    assert await store.delete(document_id) is True
    assert store.exists(document_id) is False


async def test_reading_a_missing_file_raises_a_domain_error(store):
    with pytest.raises(DocumentSourceMissingError):
        await store.read(uuid.uuid4())


async def test_deleting_a_missing_file_is_not_an_error(store):
    assert await store.delete(uuid.uuid4()) is False
    assert await store.safe_delete(uuid.uuid4()) is False


async def test_save_overwrites_and_leaves_no_partial_file(store):
    document_id = uuid.uuid4()

    await store.save(document_id, b"%PDF-1.4 first")
    await store.save(document_id, b"%PDF-1.4 second")

    assert await store.read(document_id) == b"%PDF-1.4 second"
    # The temporary write target must not survive.
    assert list(store.base_dir.glob("*.part")) == []


async def test_safe_delete_swallows_failures(store, monkeypatch):
    async def explode(_document_id):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(store, "delete", explode)

    assert await store.safe_delete(uuid.uuid4()) is False
