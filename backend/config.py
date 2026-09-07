from functools import lru_cache

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

    # --- uploads ---------------------------------------------------------
    max_upload_size_mb: int = Field(default=25, ge=1, le=500)

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
