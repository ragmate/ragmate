from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, env_file_encoding="utf-8", extra="ignore")

    LLM_MODEL: str
    LLM_API_KEY: str
    LLM_PROVIDER: Literal["openai", "anthropic", "google-genai", "mistralai", "xai", "deepseek"]
    LLM_BASE_URL: str | None = None
    LLM_TEMPERATURE: float = 0.7

    LLM_EMBEDDING_MODEL: str = "microsoft/codebert-base"
    EMBEDDING_API_KEY: str | None = None
    EMBEDDING_PROVIDER: Literal["openai", "huggingface"] = "huggingface"

    CACHE_PATH: str = "cache"
    HUGGINGFACE_MODEL_PATH: str = f"{CACHE_PATH}/huggingface"
    CHROMA_PERSIST_PATH: str = f"{CACHE_PATH}/chroma"
    PARENT_DOCSTORE_PATH: str = f"{CACHE_PATH}/docstore"
    NEW_HEAD_FILE_PATH: str = f"{CACHE_PATH}/new_head.txt"

    LOCAL_MODEL: str = "ragmate"
    REINDEX_AFTER_N_CHANGES: int = 20
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
    CUSTOM_FIRST_MESSAGE: str | None = None
    JETBRAINS_CHAT_FIRST_MESSAGE_STARTS_FROM: str = "You MUST reply in a polite and helpful manner"
    SKIP_RAG_FOR_CHAT_THAT_CONTAINS: list[str] = [
        "Answer only with yes or no.",
        "Suggest a MOST SPECIFIC title for this chat",
        "You are given a user's request to JetBrains intellij AI Assistant.",
    ]
    DEFAULT_COLLECTION_NAME: str = "collection"


@lru_cache()
def get_settings():
    return Settings()  # type: ignore[call-arg]
