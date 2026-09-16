import logging
import secrets
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Values that must never be accepted as a real signing key.
WEAK_JWT_SECRETS = frozenset(
    {"", "secret", "changeme", "change-me", "please-change-me", "dev", "test"}
)
MIN_JWT_SECRET_LENGTH = 32


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
    app_version: str = "0.4.0"
    log_level: str = "INFO"
    # "production" turns on the strict checks below. Anything else is a
    # developer machine.
    app_env: Literal["dev", "test", "production"] = "dev"

    # --- authentication --------------------------------------------------
    # Never has a usable default: production refuses to start without one, and
    # a developer machine gets a random per-process key (see the validator).
    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = Field(default=60, ge=1, le=1440)
    jwt_issuer: str = "private-ai-platform"

    password_min_length: int = Field(default=10, ge=8, le=64)
    # Argon2 hashes the whole input, but an unbounded password is a cheap DoS.
    password_max_length: int = Field(default=128, ge=64, le=1024)

    # Comma separated. Empty means "no browser origin is allowed", which is the
    # right default for an API-only service.
    cors_allowed_origins: str = ""

    # --- tracing (Langfuse) ------------------------------------------------
    # Entirely optional. With this off the application behaves exactly as it
    # did before tracing existed, and no Langfuse code runs.
    langfuse_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    # Seconds the SDK may spend on a network call before giving up. Kept short
    # because tracing must never hold a user request open.
    langfuse_timeout_seconds: int = Field(default=5, ge=1, le=60)
    # OFF by default and deliberately so: prompts, questions and retrieved
    # chunks are user content. Turning this on ships that content to whatever
    # Langfuse instance is configured. See the privacy notes in the README.
    langfuse_capture_content: bool = False
    # Truncation applied to any captured content, as a second line of defence
    # against shipping a whole document by accident.
    langfuse_max_content_chars: int = Field(default=1000, ge=0, le=20_000)
    langfuse_environment: str = "development"

    # --- rate limiting ----------------------------------------------------
    rate_limit_enabled: bool = True
    rate_limit_auth_per_minute: int = Field(default=10, ge=1, le=10_000)
    rate_limit_chat_per_minute: int = Field(default=30, ge=1, le=10_000)
    rate_limit_upload_per_minute: int = Field(default=10, ge=1, le=10_000)
    rate_limit_window_seconds: int = Field(default=60, ge=1, le=3600)

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
    # "dense"  — embeddings only (the pipeline before hybrid retrieval)
    # "hybrid" — dense + BM25 fused with Reciprocal Rank Fusion
    retrieval_mode: Literal["dense", "hybrid"] = "hybrid"

    rag_top_k: int = Field(default=5, ge=1, le=50)
    # Candidates pulled from Qdrant.
    rag_candidate_k: int = Field(default=15, ge=1, le=200)
    # Candidates pulled from the BM25 index.
    rag_lexical_candidate_k: int = Field(default=15, ge=1, le=200)
    # RRF damping: larger values flatten each list's head, so agreement between
    # the two branches counts for more than one branch's single best hit.
    rag_rrf_k: int = Field(default=60, ge=1, le=1000)
    # Upper bound on what reaches the cross-encoder — the expensive stage.
    rag_rerank_candidate_k: int = Field(default=25, ge=1, le=200)
    # Okapi BM25 parameters.
    bm25_k1: float = Field(default=1.5, ge=0.0, le=5.0)
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0)
    # Snowball stemming, chosen per token by script. Russian inflection makes
    # unstemmed BM25 nearly useless on this corpus.
    bm25_stemming: bool = True
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

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def cors_origins(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.cors_allowed_origins.split(",")
            if origin.strip()
        ]

    @model_validator(mode="after")
    def _validate_jwt_secret(self) -> "Settings":
        """Refuse a weak signing key in production; improvise one in dev.

        A hardcoded fallback would be worse than either: it would silently ship
        a publicly known key. Production fails loudly instead, and a developer
        machine gets a random key that lasts as long as the process — tokens do
        not survive a restart, which is the intended nudge to set the variable.
        """
        weak = (
            self.jwt_secret_key.strip().lower() in WEAK_JWT_SECRETS
            or len(self.jwt_secret_key) < MIN_JWT_SECRET_LENGTH
        )

        if not weak:
            return self

        if self.is_production:
            raise ValueError(
                "JWT_SECRET_KEY must be set to at least "
                f"{MIN_JWT_SECRET_LENGTH} characters when APP_ENV=production. "
                'Generate one with: python -c "import secrets; '
                'print(secrets.token_urlsafe(48))"'
            )

        object.__setattr__(self, "jwt_secret_key", secrets.token_urlsafe(48))
        logger.warning(
            "JWT_SECRET_KEY is unset or too short; using a random per-process "
            "key. Issued tokens become invalid when this process restarts."
        )

        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
