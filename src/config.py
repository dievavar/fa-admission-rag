"""Конфигурация приложения из .env без хранения секретов в коде."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    credentials: str
    scope: str
    model: str
    embedding_model: str
    base_url: str
    verify_ssl: bool
    ca_bundle_file: str
    qdrant_url: str
    collection: str
    admission_year: int
    top_k: int
    min_score: float
    data_dir: Path
    qdrant_api_key: str

    def require_credentials(self) -> None:
        if not self.credentials or self.credentials.startswith("вставьте_"):
            raise ValueError(
                "Не указан ключ GigaChat API. Скопируйте .env.example в .env "
                "и заполните GIGACHAT_CREDENTIALS."
            )


def get_settings() -> Settings:
    return Settings(
        credentials=os.getenv("GIGACHAT_CREDENTIALS", "").strip(),
        scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS").strip(),
        model=os.getenv("GIGACHAT_MODEL", "GigaChat-2-Pro").strip(),
        embedding_model=os.getenv("GIGACHAT_EMBEDDING_MODEL", "EmbeddingsGigaR").strip(),
        base_url=os.getenv("GIGACHAT_BASE_URL", "https://api.giga.chat/v1").strip(),
        verify_ssl=_env_bool("GIGACHAT_VERIFY_SSL", True),
        ca_bundle_file=os.getenv("GIGACHAT_CA_BUNDLE_FILE", "").strip(),
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333").strip(),
        collection=os.getenv("QDRANT_COLLECTION", "fa_admission_2026").strip(),
        admission_year=int(os.getenv("ADMISSION_YEAR", "2026")),
        top_k=max(1, min(10, int(os.getenv("RETRIEVAL_TOP_K", "4")))),
        min_score=float(os.getenv("MIN_SIMILARITY_SCORE", "0.0")),
        data_dir=ROOT / "data" / "raw",
        qdrant_api_key=os.getenv("QDRANT_API_KEY", "").strip(),
    )
