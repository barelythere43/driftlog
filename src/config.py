from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str
    app_env: str = "development"
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # Langfuse / OpenTelemetry tracing — keys only work for the region where the project was created.
    # EU: LANGFUSE_HOST=https://cloud.langfuse.com  |  US: LANGFUSE_HOST=https://us.cloud.langfuse.com
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = Field(
        default="https://us.cloud.langfuse.com",
        description="Langfuse base URL (must match API key region). EU cloud.langfuse.com, US us.cloud.langfuse.com",
    )

    model_config = {"env_file": ".env"}


settings = Settings()