import logging

import httpx
import redis.asyncio as redis
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from backend.config import settings
from backend.rag import RAGService


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend")


app = FastAPI(
    title="Private AI Platform API",
    version="0.2.0",
)


engine = create_async_engine(
    settings.database_url,
    pool_pre_ping=True,
)

redis_client = redis.from_url(
    settings.redis_url,
    decode_responses=True,
)

qdrant_client = AsyncQdrantClient(
    url=settings.qdrant_url,
)

rag_service = RAGService(
    qdrant_url=settings.qdrant_url
)


class AskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=5000,
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )


class Source(BaseModel):
    filename: str | None
    page: int | None
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
async def health():
    results = {
        "api": "ok",
        "postgres": "unknown",
        "redis": "unknown",
        "qdrant": "unknown",
        "inference": "unknown",
    }

    try:
        async with engine.connect() as connection:
            result = await connection.execute(
                text("SELECT 1")
            )
            result.scalar_one()

        results["postgres"] = "ok"

    except Exception as exc:
        logger.exception(
            "PostgreSQL health check failed: %s",
            exc,
        )
        results["postgres"] = "error"

    try:
        await redis_client.ping()
        results["redis"] = "ok"

    except Exception:
        results["redis"] = "error"

    try:
        await qdrant_client.get_collections()
        results["qdrant"] = "ok"

    except Exception:
        results["qdrant"] = "error"

    try:
        async with httpx.AsyncClient(
            timeout=3.0
        ) as client:
            response = await client.get(
                f"{settings.inference_url}/health"
            )
            response.raise_for_status()

        results["inference"] = "ok"

    except Exception:
        results["inference"] = "error"

    return results


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required",
        )

    if not file.filename.lower().endswith(
        ".pdf"
    ):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported",
        )

    file_bytes = await file.read()

    try:
        result = await rag_service.ingest_pdf(
            filename=file.filename,
            file_bytes=file_bytes,
        )

        return result

    except Exception as exc:
        logger.exception(
            "Document ingestion failed"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@app.post(
    "/rag/ask",
    response_model=AskResponse,
)
async def ask_rag(
    request: AskRequest,
):
    retrieved = await rag_service.retrieve(
        question=request.question,
        top_k=request.top_k,
    )

    if not retrieved:
        raise HTTPException(
            status_code=404,
            detail="No relevant documents found",
        )

    context_parts = []

    for index, item in enumerate(
        retrieved,
        start=1,
    ):
        context_parts.append(
            (
                f"[SOURCE {index}]\n"
                f"File: {item['filename']}\n"
                f"Page: {item['page']}\n"
                f"Text:\n{item['text']}"
            )
        )

    context = "\n\n".join(
        context_parts
    )

    system_prompt = (
        "Ты корпоративный AI-ассистент. "
        "Отвечай только на основе переданного контекста. "
        "Если ответа в контексте нет, прямо скажи, "
        "что информации недостаточно. "
        "Не придумывай факты. "
        "При ответе указывай источники в формате "
        "[SOURCE N]."
    )

    user_prompt = (
        f"КОНТЕКСТ:\n\n{context}\n\n"
        f"ВОПРОС:\n{request.question}"
    )

    payload = {
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "max_tokens": 500,
        "temperature": 0.1,
    }

    try:
        async with httpx.AsyncClient(
            timeout=120.0
        ) as client:
            response = await client.post(
                (
                    f"{settings.inference_url}"
                    "/v1/chat"
                ),
                headers={
                    "X-API-Key": (
                        settings.inference_api_key
                    )
                },
                json=payload,
            )

            response.raise_for_status()

            result = response.json()

    except Exception as exc:
        logger.exception(
            "Inference request failed"
        )

        raise HTTPException(
            status_code=502,
            detail="Inference service failed",
        ) from exc

    sources = [
        Source(
            filename=item["filename"],
            page=item["page"],
            score=round(
                float(item["score"]),
                4,
            ),
        )
        for item in retrieved
    ]

    return AskResponse(
        answer=result["message"]["content"],
        sources=sources,
    )
