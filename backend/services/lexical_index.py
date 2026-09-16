"""Lexical (BM25) retrieval over the chunks already stored in PostgreSQL.

Why BM25 in Python rather than a search engine
----------------------------------------------
The chunk text is already persisted in ``document_chunks`` with the tenant
reachable through ``documents.user_id``, so a lexical index needs no new
infrastructure and no second copy of the corpus. Okapi BM25 is ~40 lines and
gives an exact, inspectable ranking; PostgreSQL ``ts_rank`` would have meant
shipping a different scoring function than the one advertised.

The index is built per tenant and cached in the process. A cheap version probe
(row count plus newest chunk timestamp) rebuilds it after an ingest or a delete,
so a Celery worker writing chunks is picked up by the API process on the next
query.

Scale limit, stated plainly: the index holds one tenant's chunks in memory and
is rebuilt from scratch when the corpus changes. That suits a self-hosted
personal corpus. A large multi-tenant deployment wants Postgres FTS, OpenSearch
or Qdrant sparse vectors instead — the :class:`LexicalRetriever` seam is where
that swap would happen.
"""

import logging
import math
import re
import time
import uuid as uuid_module
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models import Document, DocumentChunk

logger = logging.getLogger(__name__)

# Unicode-aware: keeps Cyrillic and Latin words, digits and inner hyphens, so
# identifiers like "INC-2026-017" survive as one token.
_TOKEN_RE = re.compile(r"[^\W_]+(?:-[^\W_]+)*", re.UNICODE)
_CYRILLIC_RE = re.compile(r"[а-я]", re.IGNORECASE)

MIN_TOKEN_LENGTH = 2

_STEMMERS: dict[str, Any] = {}


def _stemmer(language: str) -> Any:
    if language not in _STEMMERS:
        import snowballstemmer

        _STEMMERS[language] = snowballstemmer.stemmer(language)

    return _STEMMERS[language]


def stem_token(token: str) -> str:
    """Snowball stem, picking the language from the token's own script.

    Russian is heavily inflected: without stemming, the query "проекты" simply
    does not match a chunk containing "проект", which makes BM25 decorative on
    the corpus this platform is aimed at. Per-token script detection keeps a
    bilingual corpus working without a language-detection pass over the text.

    Digits and identifiers are left alone so "INC-2026-017" stays exact.
    """
    if token.isdigit() or "-" in token:
        return token

    language = "russian" if _CYRILLIC_RE.search(token) else "english"

    return _stemmer(language).stemWord(token)


def tokenize(text: str, stemming: bool = True) -> list[str]:
    """Lowercase, normalised word tokens.

    No stop-word list: BM25's IDF already discounts terms that appear
    everywhere, so a hand-maintained list per language would add a maintenance
    burden for little gain.
    """
    if not text:
        return []

    lowered = text.lower().replace("ё", "е")

    tokens = [
        token
        for token in _TOKEN_RE.findall(lowered)
        if len(token) >= MIN_TOKEN_LENGTH or token.isdigit()
    ]

    return [stem_token(token) for token in tokens] if stemming else tokens


@dataclass(slots=True)
class LexicalDocument:
    """One indexed chunk. ``point_id`` is the join key with dense results."""

    point_id: str
    document_id: str
    chunk_index: int
    page: int | None
    filename: str
    text: str


@dataclass(slots=True)
class BM25Index:
    """Okapi BM25 over a fixed set of chunks. Pure, no I/O, deterministic."""

    documents: list[LexicalDocument]
    k1: float = 1.5
    b: float = 0.75
    stemming: bool = True

    _term_frequencies: list[Counter] = field(default_factory=list, init=False)
    _lengths: list[int] = field(default_factory=list, init=False)
    _document_frequency: dict[str, int] = field(default_factory=dict, init=False)
    _average_length: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        for document in self.documents:
            tokens = tokenize(document.text, stemming=self.stemming)
            counts = Counter(tokens)

            self._term_frequencies.append(counts)
            self._lengths.append(len(tokens))

            for term in counts:
                self._document_frequency[term] = self._document_frequency.get(term, 0) + 1

        total = sum(self._lengths)
        self._average_length = total / len(self._lengths) if self._lengths else 0.0

    def __len__(self) -> int:
        return len(self.documents)

    def _idf(self, term: str) -> float:
        """Robertson/Sparck-Jones IDF with the +1 guard against negatives."""
        n = len(self.documents)
        df = self._document_frequency.get(term, 0)

        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(
        self,
        query: str,
        limit: int,
        document_ids: set[str] | None = None,
    ) -> list[tuple[LexicalDocument, float]]:
        """Best ``limit`` chunks for the query, highest score first.

        ``document_ids`` narrows the candidate set without needing a separate
        index, so one cached index per tenant serves both scoped and unscoped
        queries.
        """
        terms = tokenize(query, stemming=self.stemming)

        if not terms or not self.documents:
            return []

        idf = {term: self._idf(term) for term in set(terms)}
        scored: list[tuple[LexicalDocument, float]] = []

        for position, document in enumerate(self.documents):
            if document_ids is not None and document.document_id not in document_ids:
                continue

            counts = self._term_frequencies[position]
            length = self._lengths[position]
            score = 0.0

            for term in terms:
                frequency = counts.get(term, 0)

                if not frequency:
                    continue

                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * (length / self._average_length or 1.0)
                )
                score += idf[term] * (frequency * (self.k1 + 1.0)) / denominator

            if score > 0.0:
                scored.append((document, score))

        # Ties broken by (document_id, chunk_index) so the order is stable
        # across runs — the evaluation harness depends on that.
        scored.sort(key=lambda item: (-item[1], item[0].document_id, item[0].chunk_index))

        return scored[:limit]


def _as_uuid(user_id: str | uuid_module.UUID) -> uuid_module.UUID:
    """Bind a real UUID to the query.

    The dense branch carries tenant ids as strings because that is what a
    Qdrant payload holds, but ``documents.user_id`` is a UUID column and
    SQLAlchemy will not coerce a string for it.
    """
    if isinstance(user_id, uuid_module.UUID):
        return user_id

    try:
        return uuid_module.UUID(str(user_id))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError(f"lexical search needs a UUID user_id, got {user_id!r}") from exc


@dataclass(slots=True)
class _CachedIndex:
    version: tuple[int, str]
    index: BM25Index
    built_at: float


class LexicalRetriever:
    """Tenant-scoped BM25 search backed by ``document_chunks``."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        stemming: bool = True,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.stemming = stemming
        self._cache: dict[str, _CachedIndex] = {}

    def invalidate(self, user_id: str | None = None) -> None:
        if user_id is None:
            self._cache.clear()
        else:
            self._cache.pop(str(user_id), None)

    async def _corpus_version(self, session: AsyncSession, user_id: str) -> tuple[int, str]:
        """Cheap change probe: (chunk count, newest chunk timestamp).

        An ingest raises both, a delete lowers the count, and a re-process
        replaces chunks with fresh timestamps — so any corpus change moves the
        tuple and forces a rebuild.
        """
        row = (
            await session.execute(
                select(
                    func.count(DocumentChunk.id),
                    func.max(DocumentChunk.created_at),
                )
                .select_from(DocumentChunk)
                .join(Document, Document.id == DocumentChunk.document_id)
                .where(Document.user_id == _as_uuid(user_id))
            )
        ).one()

        return int(row[0] or 0), str(row[1] or "")

    async def _load(self, session: AsyncSession, user_id: str) -> list[LexicalDocument]:
        result = await session.execute(
            select(
                DocumentChunk.qdrant_point_id,
                DocumentChunk.document_id,
                DocumentChunk.chunk_index,
                DocumentChunk.page,
                DocumentChunk.text,
                Document.filename,
            )
            .join(Document, Document.id == DocumentChunk.document_id)
            # The tenant boundary, enforced in SQL exactly like the dense
            # branch enforces it inside Qdrant.
            .where(Document.user_id == _as_uuid(user_id))
            .order_by(DocumentChunk.document_id, DocumentChunk.chunk_index)
        )

        return [
            LexicalDocument(
                point_id=str(row[0]),
                document_id=str(row[1]),
                chunk_index=int(row[2]),
                page=row[3],
                text=row[4] or "",
                filename=row[5] or "",
            )
            for row in result.all()
        ]

    async def index_for(self, session: AsyncSession, user_id: str) -> BM25Index:
        key = str(user_id)
        version = await self._corpus_version(session, key)
        cached = self._cache.get(key)

        if cached is not None and cached.version == version:
            return cached.index

        started = time.perf_counter()
        documents = await self._load(session, key)
        index = BM25Index(
            documents=documents,
            k1=self.k1,
            b=self.b,
            stemming=self.stemming,
        )

        self._cache[key] = _CachedIndex(
            version=version,
            index=index,
            built_at=time.time(),
        )

        logger.info(
            "lexical_index_built chunks=%d duration_ms=%.1f",
            len(index),
            (time.perf_counter() - started) * 1000,
        )

        return index

    async def search(
        self,
        session: AsyncSession,
        question: str,
        user_id: str,
        limit: int,
        document_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return hits in the same dict shape the dense branch produces."""
        if not user_id:
            raise ValueError("lexical search requires a user_id")

        index = await self.index_for(session, user_id)
        allowed = {str(value) for value in document_ids} if document_ids else None
        hits = index.search(question, limit=limit, document_ids=allowed)

        return [
            {
                "point_id": document.point_id,
                "user_id": str(user_id),
                "document_id": document.document_id,
                "filename": document.filename,
                "page": document.page,
                "chunk_index": document.chunk_index,
                "text": document.text,
                "lexical_score": round(float(score), 6),
                "lexical_rank": rank,
            }
            for rank, (document, score) in enumerate(hits, start=1)
        ]
