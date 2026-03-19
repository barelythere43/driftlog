from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str
    app_env: str = "development"
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # Langfuse / OpenTelemetry tracing
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    model_config = {"env_file": ".env"}


settings = Settings()