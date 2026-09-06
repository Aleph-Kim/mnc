from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    storage_dir: str = "storage"
    max_upload_mb: int = 20

    # 파이프라인 단계별 중간 이미지를 outputs/<id>/debug/에 덤프 (구조 개편 진단용)
    debug_pipeline: bool = True

    class Config:
        env_file = ".env"


settings = Settings()
