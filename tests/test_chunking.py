import pytest

from backend.errors import InvalidDocumentError
from backend.services.chunking import build_chunks, chunk_text, extract_pdf_pages


def test_chunk_text_short_text_is_a_single_chunk():
    assert chunk_text("два слова", chunk_size=10, overlap=2) == ["два слова"]


def test_chunk_text_empty_text_produces_no_chunks():
    assert chunk_text("   ", chunk_size=10, overlap=2) == []


def test_chunk_text_splits_with_overlap():
    words = [f"w{index}" for index in range(25)]
    chunks = chunk_text(" ".join(words), chunk_size=10, overlap=3)

    assert len(chunks) == 4
    assert chunks[0].split() == words[:10]
    # The second window restarts three words before the first one ended.
    assert chunks[1].split()[0] == "w7"
    assert chunks[-1].split()[-1] == "w24"


def test_chunk_text_covers_every_word():
    words = [f"w{index}" for index in range(97)]
    chunks = chunk_text(" ".join(words), chunk_size=20, overlap=5)

    covered = {word for chunk in chunks for word in chunk.split()}
    assert covered == set(words)


def test_chunk_text_overlap_larger_than_window_does_not_hang():
    chunks = chunk_text(" ".join(str(index) for index in range(50)), chunk_size=5, overlap=99)

    assert len(chunks) > 1
    assert chunks[-1].split()[-1] == "49"


@pytest.mark.parametrize("chunk_size,overlap", [(0, 0), (-1, 0)])
def test_chunk_text_rejects_invalid_window(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=chunk_size, overlap=overlap)


def test_chunk_text_rejects_negative_overlap():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=10, overlap=-1)


def test_build_chunks_numbers_chunks_across_pages():
    pages = [
        {"page": 1, "text": " ".join(f"a{index}" for index in range(30))},
        {"page": 2, "text": " ".join(f"b{index}" for index in range(30))},
    ]

    records = build_chunks(pages, chunk_size=10, overlap=2)

    assert [record["chunk_index"] for record in records] == list(range(len(records)))
    assert {record["page"] for record in records} == {1, 2}
    assert all(record["text"] for record in records)


def test_build_chunks_of_empty_pages_is_empty():
    assert build_chunks([], chunk_size=10, overlap=2) == []


def test_extract_pdf_pages_rejects_non_pdf_bytes():
    with pytest.raises(InvalidDocumentError):
        extract_pdf_pages(b"this is definitely not a pdf")
