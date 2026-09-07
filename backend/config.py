from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings.

    Every field maps to an upper-case environment variable of the same name
    (``rag_top_k`` -> ``RAG_TOP_K``) and can be overridden through ``.env``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- application ---------------------------------------------------
    app_name: str = "Private AI Platform API"
    app_version: str = "0.3.0"
    log_level: str = "INFO"

    # --- infrastructure ------------------------------------------------
    database_url: str = (
        "postgresql+asyncpg://privateai:privateai_dev_password@127.0.0.1:5432/privateai"
    )
    redis_url: str = "redis://127.0.0.1:6379/0"
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "documents"

    # --- inference service ---------------------------------------------
    inference_url: str = "http://127.0.0.1:8001"
    # Never commit a real key: supply it through .env / the environment.
    inference_api_key: str = ""
    inference_timeout_seconds: float = Field(default=120.0, gt=0)
    inference_health_timeout_seconds: float = Field(default=3.0, gt=0)
    inference_max_tokens: int = Field(default=500, ge=1, le=2000)
    inference_temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    # --- models ---------------------------------------------------------
    embedding_model: str = "intfloat/multilingual-e5-small"
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    embedding_dim: int = Field(default=384, ge=1)
    # Load the embedding / reranker models during startup instead of on the
    # first request that needs them.
    preload_models: bool = True

    # --- RAG ------------------------------------------------------------
    rag_top_k: int = Field(default=5, ge=1, le=50)
    rag_candidate_k: int = Field(default=15, ge=1, le=200)
    chunk_size_words: int = Field(default=220, ge=20)
    chunk_overlap_words: int = Field(default=40, ge=0)
    # Hard cap on the grounded context handed to the LLM. The inference
    # service rejects individual messages longer than 20k characters.
    max_context_chars: int = Field(default=12_000, ge=500, le=18_000)

    # --- chat -----------------------------------------------------------
    chat_history_limit: int = Field(default=20, ge=1, le=200)

    # --- agent -----------------------------------------------------------
    # Hard ceiling on graph nodes executed per request. The graph is a fixed
    # DAG (classify -> branch -> compose), so a normal run uses 3; the limit
    # exists so that future extensions can never turn into a runaway loop.
    agent_max_steps: int = Field(default=6, ge=1, le=50)
    # Structured calls (routing, tool selection) want deterministic output.
    agent_router_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    agent_structured_max_tokens: int = Field(default=300, ge=32, le=2000)
    agent_answer_max_tokens: int = Field(default=600, ge=32, le=2000)
    # A structured call is attempted once, then repaired at most once.
    agent_structured_repair_attempts: int = Field(default=1, ge=0, le=2)

    # --- uploads ---------------------------------------------------------
    max_upload_size_mb: int = Field(default=25, ge=1, le=500)
    # Where uploaded PDFs live until a worker has ingested them. Relative paths
    # are resolved against the process working directory.
    upload_dir: Path = Path("data/uploads")
    # Drop the source PDF once its chunks are indexed. Turn off to keep the
    # originals for re-processing.
    delete_source_after_processing: bool = True

    # --- Celery / RabbitMQ -----------------------------------------------
    celery_broker_url: str = (
        "amqp://privateai:privateai_dev_password@127.0.0.1:5672//"
    )
    celery_result_backend: str = "redis://127.0.0.1:6379/1"
    celery_task_queue: str = "documents"
    # Hard/soft limits for a single ingestion task.
    celery_task_soft_time_limit: int = Field(default=1500, ge=30)
    celery_task_time_limit: int = Field(default=1800, ge=60)
    # Retry policy for transient failures (Qdrant or PostgreSQL hiccups).
    celery_max_retries: int = Field(default=5, ge=0, le=20)
    celery_retry_backoff_seconds: int = Field(default=5, ge=1)
    celery_retry_backoff_max_seconds: int = Field(default=300, ge=1)
    # Run tasks inline instead of dispatching them. Tests only.
    celery_task_always_eager: bool = False
    broker_health_timeout_seconds: float = Field(default=2.0, gt=0)
    worker_ping_timeout_seconds: float = Field(default=1.0, gt=0)
    # 0 disables the worker's Prometheus HTTP server. Only meaningful for a
    # single-process pool (solo/threads); see README.
    worker_metrics_port: int = Field(default=0, ge=0, le=65535)

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
