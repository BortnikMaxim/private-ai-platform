import logging
import os
import time
import uuid

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from prometheus_client import Counter, Histogram, generate_latest

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

API_KEY = os.getenv("INFERENCE_API_KEY", "dev-secret")


REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total inference API requests",
    ["endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "inference_request_duration_seconds",
    "Inference request duration",
    ["endpoint"],
)

GENERATION_LATENCY = Histogram(
    "llm_generation_duration_seconds",
    "LLM generation duration",
)


app = FastAPI(
    title="Private AI Inference Service",
    version="0.4.0",
)


def verify_api_key(x_api_key: str | None) -> None:
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id

    start = time.perf_counter()

    try:
        response = await call_next(request)

        duration = time.perf_counter() - start

        response.headers["X-Request-ID"] = request_id

        logger.info(
            "request_completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration * 1000, 2),
            },
        )

        return response

    except Exception:
        logger.exception(
            "request_failed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
            },
        )
        raise


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
    }


@app.get("/ready")
def ready():
    return {
        "status": "ready",
        "model": MODEL_ID,
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return generate_latest().decode("utf-8")


@app.post("/v1/generate", response_model=GenerateResponse)
def generate_text(
    request: GenerateRequest,
    x_api_key: str | None = Header(default=None),
):
    verify_api_key(x_api_key)

    start = time.perf_counter()

    try:
        text, elapsed = gemma_service.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        GENERATION_LATENCY.observe(elapsed)
        REQUEST_LATENCY.labels(endpoint="/v1/generate").observe(
            time.perf_counter() - start
        )
        REQUEST_COUNT.labels(
            endpoint="/v1/generate",
            status="success",
        ).inc()

        return GenerateResponse(
            model=MODEL_ID,
            text=text,
            generation_time_seconds=round(elapsed, 3),
        )

    except Exception as exc:
        REQUEST_COUNT.labels(
            endpoint="/v1/generate",
            status="error",
        ).inc()

        raise HTTPException(
            status_code=500,
            detail="Model generation failed",
        ) from exc


@app.post("/v1/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    x_api_key: str | None = Header(default=None),
):
    verify_api_key(x_api_key)

    messages = [
        {
            "role": message.role,
            "content": message.content,
        }
        for message in request.messages
    ]

    start = time.perf_counter()

    try:
        text, elapsed = gemma_service.chat(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        GENERATION_LATENCY.observe(elapsed)
        REQUEST_LATENCY.labels(endpoint="/v1/chat").observe(
            time.perf_counter() - start
        )
        REQUEST_COUNT.labels(
            endpoint="/v1/chat",
            status="success",
        ).inc()

        return ChatResponse(
            model=MODEL_ID,
            message=ChatMessage(
                role="assistant",
                content=text,
            ),
            generation_time_seconds=round(elapsed, 3),
        )

    except Exception as exc:
        REQUEST_COUNT.labels(
            endpoint="/v1/chat",
            status="error",
        ).inc()

        raise HTTPException(
            status_code=500,
            detail="Chat generation failed",
        ) from exc


@app.post("/v1/chat/stream")
def chat_stream(
    request: ChatRequest,
    x_api_key: str | None = Header(default=None),
):
    verify_api_key(x_api_key)

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
            GENERATION_LATENCY.observe(elapsed)

            REQUEST_COUNT.labels(
                endpoint="/v1/chat/stream",
                status="success",
            ).inc()

        except Exception:
            REQUEST_COUNT.labels(
                endpoint="/v1/chat/stream",
                status="error",
            ).inc()
            raise

    return StreamingResponse(
        token_stream(),
        media_type="text/plain",
    )
