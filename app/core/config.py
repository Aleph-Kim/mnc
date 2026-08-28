from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    storage_dir: str = "storage"
    max_upload_mb: int = 20

    class Config:
        env_file = ".env"


settings = Settings()
