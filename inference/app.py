from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from inference.model import MODEL_ID, gemma_service
from inference.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    GenerateRequest,
    GenerateResponse,
)


app = FastAPI(
    title="Private AI Inference Service",
    version="0.2.0",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
    }


@app.post("/v1/generate", response_model=GenerateResponse)
def generate_text(request: GenerateRequest):
    try:
        text, elapsed = gemma_service.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        return GenerateResponse(
            model=MODEL_ID,
            text=text,
            generation_time_seconds=round(elapsed, 3),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Model generation failed",
        ) from exc


@app.post("/v1/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
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

        return ChatResponse(
            model=MODEL_ID,
            message=ChatMessage(
                role="assistant",
                content=text,
            ),
            generation_time_seconds=round(elapsed, 3),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Chat generation failed",
        ) from exc


@app.post("/v1/chat/stream")
def chat_stream(request: ChatRequest):
    messages = [
        {
            "role": message.role,
            "content": message.content,
        }
        for message in request.messages
    ]

    def token_stream():
        for chunk in gemma_service.stream_chat(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        ):
            yield chunk

    return StreamingResponse(
        token_stream(),
        media_type="text/plain",
    )
