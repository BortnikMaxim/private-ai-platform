import asyncio
import io
import uuid

from pypdf import PdfReader
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer


COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
VECTOR_SIZE = 384


class RAGService:
    def __init__(
        self,
        qdrant_url: str,
    ) -> None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}")

        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        self.qdrant = AsyncQdrantClient(
            url=qdrant_url
        )

        print("Embedding model loaded successfully")

    async def ensure_collection(self) -> None:
        exists = await self.qdrant.collection_exists(
            COLLECTION_NAME
        )

        if not exists:
            await self.qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )

    def extract_pdf(
        self,
        file_bytes: bytes,
    ) -> list[dict]:
        reader = PdfReader(
            io.BytesIO(file_bytes)
        )

        pages = []

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):
            text = page.extract_text()

            if text and text.strip():
                pages.append(
                    {
                        "page": page_number,
                        "text": text.strip(),
                    }
                )

        return pages

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 220,
        overlap: int = 40,
    ) -> list[str]:
        words = text.split()

        chunks = []

        start = 0

        while start < len(words):
            end = start + chunk_size

            chunk = " ".join(
                words[start:end]
            )

            if chunk.strip():
                chunks.append(chunk)

            if end >= len(words):
                break

            start = end - overlap

        return chunks

    def embed_passages(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        passages = [
            f"passage: {text}"
            for text in texts
        ]

        embeddings = self.embedding_model.encode(
            passages,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embeddings.tolist()

    def embed_query(
        self,
        question: str,
    ) -> list[float]:
        embedding = self.embedding_model.encode(
            f"query: {question}",
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embedding.tolist()

    async def ingest_pdf(
        self,
        filename: str,
        file_bytes: bytes,
    ) -> dict:
        await self.ensure_collection()

        document_id = str(uuid.uuid4())

        pages = await asyncio.to_thread(
            self.extract_pdf,
            file_bytes,
        )

        all_chunks = []

        for page in pages:
            chunks = self.chunk_text(
                page["text"]
            )

            for chunk_index, chunk in enumerate(chunks):
                all_chunks.append(
                    {
                        "page": page["page"],
                        "chunk_index": chunk_index,
                        "text": chunk,
                    }
                )

        if not all_chunks:
            raise ValueError(
                "No text could be extracted from PDF"
            )

        texts = [
            item["text"]
            for item in all_chunks
        ]

        embeddings = await asyncio.to_thread(
            self.embed_passages,
            texts,
        )

        points = []

        for chunk, embedding in zip(
            all_chunks,
            embeddings,
        ):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "document_id": document_id,
                        "filename": filename,
                        "page": chunk["page"],
                        "chunk_index": chunk["chunk_index"],
                        "text": chunk["text"],
                    },
                )
            )

        await self.qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

        return {
            "document_id": document_id,
            "filename": filename,
            "pages": len(pages),
            "chunks": len(points),
        }

    async def retrieve(
        self,
        question: str,
        top_k: int = 5,
    ) -> list[dict]:
        await self.ensure_collection()

        query_vector = await asyncio.to_thread(
            self.embed_query,
            question,
        )

        result = await self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        retrieved = []

        for point in result.points:
            payload = point.payload or {}

            retrieved.append(
                {
                    "score": point.score,
                    "document_id": payload.get(
                        "document_id"
                    ),
                    "filename": payload.get(
                        "filename"
                    ),
                    "page": payload.get("page"),
                    "chunk_index": payload.get(
                        "chunk_index"
                    ),
                    "text": payload.get("text"),
                }
            )

        return retrieved
