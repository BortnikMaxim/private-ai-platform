"""Qdrant access layer.

Everything the application does with vectors goes through this class so the
payload shape and the collection lifecycle live in exactly one place.
"""

import logging
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

logger = logging.getLogger(__name__)

# Payload keys written for every chunk. user_id is the tenant boundary.
PAYLOAD_KEYS = (
    "user_id",
    "document_id",
    "filename",
    "page",
    "chunk_index",
    "text",
)

# Fields we filter on, so Qdrant can index them.
INDEXED_PAYLOAD_FIELDS = ("user_id", "document_id")


class VectorStore:
    def __init__(
        self,
        client: AsyncQdrantClient,
        collection_name: str,
        vector_size: int,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.vector_size = vector_size

    # -- lifecycle -------------------------------------------------------

    async def ensure_collection(self) -> None:
        if not await self.client.collection_exists(self.collection_name):
            logger.info("creating qdrant collection: %s", self.collection_name)
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

        # Every query filters on user_id, and scoped queries add document_id.
        for field in INDEXED_PAYLOAD_FIELDS:
            try:
                await self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:  # noqa: BLE001 - index already exists is not fatal
                logger.debug("%s payload index already present", field)

    async def health(self) -> bool:
        try:
            await self.client.get_collections()
            return True
        except Exception:  # noqa: BLE001 - health checks never raise
            return False

    async def aclose(self) -> None:
        await self.client.close()

    # -- writes ----------------------------------------------------------

    async def upsert_chunks(self, records: list[dict[str, Any]]) -> None:
        """Upsert chunk records.

        Each record must carry ``point_id``, ``vector``, ``user_id`` and the
        remaining payload keys. ``user_id`` is required rather than optional so
        an untagged point — which no tenant-scoped query could ever return —
        cannot be written by accident.
        """
        if not records:
            return

        points = []

        for record in records:
            user_id = record.get("user_id")

            if not user_id:
                raise ValueError("refusing to index a chunk without a user_id")

            points.append(
                PointStruct(
                    id=record["point_id"],
                    vector=record["vector"],
                    payload={
                        "user_id": str(user_id),
                        "document_id": str(record["document_id"]),
                        "filename": record["filename"],
                        "page": record.get("page"),
                        "chunk_index": record["chunk_index"],
                        "text": record["text"],
                    },
                )
            )

        await self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    async def delete_document(
        self,
        document_id: str,
        user_id: str | None = None,
    ) -> None:
        """Delete a document's points, optionally constrained to one tenant."""
        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(
                filter=build_filter(user_id=user_id, document_ids=[document_id])
            ),
            wait=True,
        )

    # -- reads -----------------------------------------------------------

    async def search(
        self,
        vector: list[float],
        limit: int,
        user_id: str,
        document_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Nearest neighbours **within one tenant**.

        ``user_id`` is a required positional-by-name argument, and the filter is
        applied by Qdrant during the search — not by Python afterwards. Post
        filtering would be both wrong (a foreign chunk that outranks everything
        would silently shrink the result set) and unsafe (the data would have
        been read already).
        """
        if not user_id:
            raise ValueError("search requires a user_id")

        response = await self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit,
            with_payload=True,
            query_filter=build_filter(user_id=user_id, document_ids=document_ids),
        )

        results: list[dict[str, Any]] = []

        for point in response.points:
            payload = point.payload or {}
            score = float(point.score)

            results.append(
                {
                    "score": score,
                    "vector_score": score,
                    "user_id": payload.get("user_id"),
                    "document_id": payload.get("document_id"),
                    "filename": payload.get("filename"),
                    "page": payload.get("page"),
                    "chunk_index": payload.get("chunk_index"),
                    "text": payload.get("text"),
                }
            )

        return results


def build_filter(
    user_id: str | None = None,
    document_ids: list[str] | None = None,
) -> Filter | None:
    """Server-side filter: tenant first, then an optional document scope."""
    conditions = []

    if user_id:
        conditions.append(
            FieldCondition(key="user_id", match=MatchValue(value=str(user_id)))
        )

    if document_ids:
        conditions.append(
            FieldCondition(
                key="document_id",
                match=MatchAny(any=[str(value) for value in document_ids]),
            )
        )

    return Filter(must=conditions) if conditions else None
