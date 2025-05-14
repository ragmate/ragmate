from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, env_file_encoding="utf-8", extra="ignore")

    LLM_MODEL: str
    LLM_API_KEY: str
    LLM_PROVIDER: Literal["openai", "anthropic", "google-genai", "mistralai", "xai", "deepseek"]

    LLM_EMBEDDING_MODEL: str = "all‑MiniLM‑L6‑v2.gguf2.f16.gguf"
    EMBEDDING_API_KEY: str | None = None
    EMBEDDING_PROVIDER: Literal["openai", "gpt4all"] = "gpt4all"

    CACHE_PATH: str = "cache"
    WATCHED_FILES_STORE_PATH: str = f"{CACHE_PATH}/watched_files.json"
    VECTOR_STORE_DUMP_PATH: str = f"{CACHE_PATH}/vector_store.json"
    GPT4ALL_MODEL_PATH: str = f"{CACHE_PATH}/gpt4all"

    LOCAL_MODEL: str = "ragmate"
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
