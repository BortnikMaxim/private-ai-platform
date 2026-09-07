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
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

logger = logging.getLogger(__name__)

# Payload keys written for every chunk.
PAYLOAD_KEYS = ("document_id", "filename", "page", "chunk_index", "text")


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

        # Filtering by document_id is on the hot path for scoped RAG queries.
        try:
            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="document_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:  # noqa: BLE001 - index already exists is not fatal
            logger.debug("document_id payload index already present")

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

        Each record must carry ``point_id``, ``vector`` and the payload keys.
        """
        if not records:
            return

        points = [
            PointStruct(
                id=record["point_id"],
                vector=record["vector"],
                payload={
                    "document_id": record["document_id"],
                    "filename": record["filename"],
                    "page": record.get("page"),
                    "chunk_index": record["chunk_index"],
                    "text": record["text"],
                },
            )
            for record in records
        ]

        await self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    async def delete_document(self, document_id: str) -> None:
        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(filter=_document_filter([document_id])),
            wait=True,
        )

    # -- reads -----------------------------------------------------------

    async def search(
        self,
        vector: list[float],
        limit: int,
        document_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        response = await self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit,
            with_payload=True,
            query_filter=_document_filter(document_ids) if document_ids else None,
        )

        results: list[dict[str, Any]] = []

        for point in response.points:
            payload = point.payload or {}
            score = float(point.score)

            results.append(
                {
                    "score": score,
                    "vector_score": score,
                    "document_id": payload.get("document_id"),
                    "filename": payload.get("filename"),
                    "page": payload.get("page"),
                    "chunk_index": payload.get("chunk_index"),
                    "text": payload.get("text"),
                }
            )

        return results


def _document_filter(document_ids: list[str]) -> Filter:
    return Filter(
        must=[
            FieldCondition(
                key="document_id",
                match=MatchAny(any=list(document_ids)),
            )
        ]
    )
