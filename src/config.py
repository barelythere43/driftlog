from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str
    app_env: str = "development"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    model_config = {"env_file": ".env"}


settings = Settings()