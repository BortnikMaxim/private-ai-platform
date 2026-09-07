"""Pure text processing helpers: PDF extraction and word based chunking.

Kept free of models, network and database access so it can be unit tested
without any external dependency.
"""

import io
from typing import Any

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from backend.errors import InvalidDocumentError


def extract_pdf_pages(file_bytes: bytes) -> tuple[int, list[dict[str, Any]]]:
    """Return ``(total_pages, pages_with_text)``.

    Pages without extractable text are skipped, which is why the returned list
    can be shorter than ``total_pages``.
    """
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        total_pages = len(reader.pages)
        pages: list[dict[str, Any]] = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()

            if text and text.strip():
                pages.append({"page": page_number, "text": text.strip()})

    except InvalidDocumentError:
        raise
    except (PyPdfError, ValueError, OSError) as exc:
        raise InvalidDocumentError("File could not be parsed as a PDF") from exc

    return total_pages, pages


def chunk_text(
    text: str,
    chunk_size: int = 220,
    overlap: int = 40,
) -> list[str]:
    """Split text into overlapping word windows."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")

    # A window that never advances would loop forever.
    overlap = min(overlap, chunk_size - 1)

    words = text.split()

    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def build_chunks(
    pages: list[dict[str, Any]],
    chunk_size: int = 220,
    overlap: int = 40,
) -> list[dict[str, Any]]:
    """Flatten extracted pages into a list of chunk records.

    ``chunk_index`` is global across the document so that chunks stay ordered
    and uniquely addressable when they are stored in PostgreSQL and Qdrant.
    """
    records: list[dict[str, Any]] = []
    chunk_index = 0

    for page in pages:
        for chunk in chunk_text(page["text"], chunk_size=chunk_size, overlap=overlap):
            records.append(
                {
                    "page": page.get("page"),
                    "chunk_index": chunk_index,
                    "text": chunk,
                }
            )
            chunk_index += 1

    return records
