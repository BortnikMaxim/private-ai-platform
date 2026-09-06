import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from inference.logging_config import setup_logging
from inference.model import MODEL_ID, gemma_service
from inference.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    GenerateRequest,
    GenerateResponse,
)


setup_logging()

logger = logging.getLogger("inference_api")


app = FastAPI(
    title="Private AI Inference Service",
    version="0.3.0",
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

    request.state.request_id = request_id

    start = time.perf_counter()

    logger.info(
        "request_started",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
        },
    )

    try:
        response = await call_next(request)

        duration_ms = round(
            (time.perf_counter() - start) * 1000,
            2,
        )

        response.headers["X-Request-ID"] = request_id

        logger.info(
            "request_completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )

        return response

    except Exception:
        duration_ms = round(
            (time.perf_counter() - start) * 1000,
            2,
        )

        logger.exception(
            "request_failed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": 500,
                "duration_ms": duration_ms,
            },
        )

        raise


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
    }


@app.post("/v1/generate", response_model=GenerateResponse)
def generate_text(request: GenerateRequest, http_request: Request):
    request_id = http_request.state.request_id

    try:
        text, elapsed = gemma_service.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        logger.info(
            "generation_completed",
            extra={
                "request_id": request_id,
                "generation_time_seconds": round(elapsed, 3),
            },
        )

        return GenerateResponse(
            model=MODEL_ID,
            text=text,
            generation_time_seconds=round(elapsed, 3),
        )

    except Exception as exc:
        logger.exception(
            "generation_failed",
            extra={
                "request_id": request_id,
            },
        )

        raise HTTPException(
            status_code=500,
            detail="Model generation failed",
        ) from exc


@app.post("/v1/chat", response_model=ChatResponse)
def chat(request: ChatRequest, http_request: Request):
    request_id = http_request.state.request_id

    try:
        messages = [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in request.messages
        ]

        text, elapsed = gemma_service.chat(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        logger.info(
            "chat_completed",
            extra={
                "request_id": request_id,
                "generation_time_seconds": round(elapsed, 3),
            },
        )

        return ChatResponse(
            model=MODEL_ID,
            message=ChatMessage(
                role="assistant",
                content=text,
            ),
            generation_time_seconds=round(elapsed, 3),
        )

    except Exception as exc:
        logger.exception(
            "chat_failed",
            extra={
                "request_id": request_id,
            },
        )

        raise HTTPException(
            status_code=500,
            detail="Chat generation failed",
        ) from exc


@app.post("/v1/chat/stream")
def chat_stream(request: ChatRequest, http_request: Request):
    request_id = http_request.state.request_id

    messages = [
        {
            "role": message.role,
            "content": message.content,
        }
        for message in request.messages
    ]

    def token_stream():
        start = time.perf_counter()

        try:
            for chunk in gemma_service.stream_chat(
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            ):
                yield chunk

            elapsed = time.perf_counter() - start

            logger.info(
                "stream_completed",
                extra={
                    "request_id": request_id,
                    "generation_time_seconds": round(elapsed, 3),
                },
            )

        except Exception:
            logger.exception(
                "stream_failed",
                extra={
                    "request_id": request_id,
                },
            )

            raise

    return StreamingResponse(
        token_stream(),
        media_type="text/plain",
        headers={
            "X-Request-ID": request_id,
        },
    )
