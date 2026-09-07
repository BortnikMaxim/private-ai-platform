"""Maintenance utility for the application's Qdrant collection.

This is a manually invoked developer tool. Nothing here runs on application
startup, and every destructive mode requires an explicit --yes.

    # 1. Report only (safe, the default)
    python -m backend.scripts.reset_qdrant

    # 2. Delete only vectors whose document_id has no row in PostgreSQL
    python -m backend.scripts.reset_qdrant --purge-orphans --yes

    # 3. Drop and recreate the whole collection
    python -m backend.scripts.reset_qdrant --recreate --yes

Only the collection named by QDRANT_COLLECTION is ever touched.
"""

import argparse
import asyncio
import sys

from qdrant_client import AsyncQdrantClient
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from backend.config import get_settings
from backend.db import create_engine
from backend.models import Document
from backend.services.vector_store import VectorStore


async def _known_document_ids(database_url: str) -> set[str] | None:
    """Document ids known to PostgreSQL, or None if they cannot be read.

    Returning None (rather than raising) keeps the report usable on a database
    that has not been migrated yet — which is exactly when orphan vectors from
    an older schema are most likely to exist.
    """
    engine = create_engine(database_url)

    try:
        async with engine.connect() as connection:
            result = await connection.execute(select(Document.id))
            return {str(value) for value in result.scalars().all()}
    except (SQLAlchemyError, OSError) as exc:
        reason = type(exc).__name__
        if "UndefinedTable" in str(exc):
            reason = "the 'documents' table does not exist (run `alembic upgrade head`)"
        print(f"WARNING: cannot read documents from PostgreSQL: {reason}")
        return None
    finally:
        await engine.dispose()


async def _scroll_document_ids(
    client: AsyncQdrantClient,
    collection: str,
) -> dict[str, int]:
    """Count points per document_id by scrolling the whole collection."""
    counts: dict[str, int] = {}
    offset = None

    while True:
        points, offset = await client.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=["document_id"],
            with_vectors=False,
        )

        for point in points:
            document_id = str((point.payload or {}).get("document_id"))
            counts[document_id] = counts.get(document_id, 0) + 1

        if offset is None:
            break

    return counts


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    collection = settings.qdrant_collection

    client = AsyncQdrantClient(url=settings.qdrant_url)
    store = VectorStore(
        client=client,
        collection_name=collection,
        vector_size=settings.embedding_dim,
    )

    try:
        try:
            exists = await client.collection_exists(collection)
        except Exception as exc:  # noqa: BLE001 - report, do not dump a traceback
            print(f"ERROR: Qdrant is not reachable at {settings.qdrant_url}: {exc}")
            return 2

        if not exists:
            print(f"Collection '{collection}' does not exist at {settings.qdrant_url}.")

            if args.recreate and args.yes:
                await store.ensure_collection()
                print(f"Created '{collection}' (dim={settings.embedding_dim}, cosine).")

            return 0

        info = await client.get_collection(collection)
        counts = await _scroll_document_ids(client, collection)
        known = await _known_document_ids(settings.database_url)

        orphans: dict[str, int] = {}
        if known is not None:
            orphans = {
                document_id: count
                for document_id, count in counts.items()
                if document_id not in known
            }

        print(f"Qdrant     : {settings.qdrant_url}")
        print(f"Collection : {collection}")
        print(f"Points     : {info.points_count}")

        if known is None:
            print(f"Documents  : {len(counts)} in Qdrant, unknown in PostgreSQL")
            print("Orphans    : cannot be determined without the documents table")
        else:
            print(f"Documents  : {len(counts)} in Qdrant, {len(known)} in PostgreSQL")
            print(
                f"Orphans    : {len(orphans)} document(s), "
                f"{sum(orphans.values())} point(s) with no Document row"
            )

            for document_id, count in sorted(orphans.items()):
                print(f"  - {document_id}: {count} point(s)")

        if not (args.purge_orphans or args.recreate):
            print("\nNothing changed. Pass --purge-orphans or --recreate (with --yes).")
            return 0

        if args.purge_orphans and known is None:
            print(
                "\nRefusing to purge: without the documents table every vector "
                "would look orphaned. Run `alembic upgrade head` first, or use "
                "--recreate to empty the collection."
            )
            return 2

        action = "recreate the collection" if args.recreate else "purge orphan vectors"

        if not args.yes:
            print(f"\nWould {action}. Re-run with --yes to actually do it.")
            return 1

        if args.recreate:
            await client.delete_collection(collection)
            await store.ensure_collection()
            print(
                f"\nRecreated '{collection}' "
                f"(dim={settings.embedding_dim}, cosine). All vectors were removed."
            )
            return 0

        if not orphans:
            print("\nNo orphan vectors to purge.")
            return 0

        for document_id in orphans:
            await store.delete_document(document_id)

        remaining = await client.get_collection(collection)
        print(
            f"\nPurged {sum(orphans.values())} orphan point(s). "
            f"Points remaining: {remaining.points_count}"
        )
        return 0

    finally:
        await client.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backend.scripts.reset_qdrant",
        description=(
            "Inspect, purge or recreate the application's Qdrant collection. "
            "Without --purge-orphans or --recreate it only reports."
        ),
    )
    parser.add_argument(
        "--purge-orphans",
        action="store_true",
        help="delete vectors whose document_id has no matching Document row",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="drop the collection and recreate it empty with the configured vector size",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="confirm a destructive action; without it nothing is modified",
    )

    args = parser.parse_args()

    if args.purge_orphans and args.recreate:
        parser.error("--purge-orphans and --recreate are mutually exclusive")

    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
