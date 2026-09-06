from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = (
        "postgresql+asyncpg://privateai:"
        "privateai_dev_password@127.0.0.1:5432/privateai"
    )

    redis_url: str = "redis://127.0.0.1:6379/0"

    qdrant_url: str = "http://127.0.0.1:6333"

    inference_url: str = "http://127.0.0.1:8001"

    inference_api_key: str = "super-secret-test-key"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
