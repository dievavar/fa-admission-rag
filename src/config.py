"""Параметры через .env. Разные ключи для Cloud.ru и GigaChat."""
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
    embedding_provider: str
    cloud_ru_api_key: str
    cloud_ru_embedding_model: str
    cloud_ru_base_url: str

    def require_credentials(self) -> None:
        if not self.credentials or self.credentials.startswith("вставьте_"):
            raise ValueError("Не заполнен GIGACHAT_CREDENTIALS в .env")

    def require_cloud_ru_credentials(self) -> None:
        if not self.cloud_ru_api_key or self.cloud_ru_api_key.startswith("вставьте_"):
            raise ValueError(
                "Не заполнен CLOUD_RU_API_KEY в .env. "
                "Создайте API Key для Foundation Models в Cloud.ru."
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
        collection=os.getenv("QDRANT_COLLECTION", "fa_admission_2026_cloudru_3b").strip(),
        admission_year=int(os.getenv("ADMISSION_YEAR", "2026")),
        top_k=max(1, min(10, int(os.getenv("RETRIEVAL_TOP_K", "4")))),
        min_score=float(os.getenv("MIN_SIMILARITY_SCORE", "0.0")),
        data_dir=ROOT / "data" / "raw",
        qdrant_api_key=os.getenv("QDRANT_API_KEY", "").strip(),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "cloudru").strip().lower(),
        cloud_ru_api_key=os.getenv("CLOUD_RU_API_KEY", "").strip(),
        cloud_ru_embedding_model=os.getenv("CLOUD_RU_EMBEDDING_MODEL", "ai-sage/Giga-Embeddings-instruct-3B").strip(),
        cloud_ru_base_url=os.getenv("CLOUD_RU_BASE_URL", "https://foundation-models.api.cloud.ru/v1").strip(),
    )
