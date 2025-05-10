from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, env_file_encoding="utf-8", extra="ignore")

    LLM_MODEL: str
    LLM_EMBEDDING_MODEL: str
    LLM_API_KEY: str
    LOCAL_MODEL: str = "ragmate"
    WATCHED_FILES_STORE_PATH: str = "cache/watched_files.json"
    VECTOR_STORE_DUMP_PATH: str = "cache/vector_store.json"
    REINDEX_AFTER_N_CHANGES: int = 100
    PROJECT_PATH: str = "/project"
    FRAMEWORK: str | None = None
    TEXT_FILE_EXTENSIONS: list[str] = [
        ".py",
        ".js",
        ".ts",
        ".php",
        ".java",
        ".rb",
        ".go",
        ".cs",
        ".rs",
        ".html",
        ".css",
    ]


@lru_cache()
def get_settings():
    return Settings()  # type: ignore[call-arg]
