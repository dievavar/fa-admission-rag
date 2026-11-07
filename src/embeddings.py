"""Векторизация через Cloud.ru Foundation Models API или старый GigaChat API."""
from __future__ import annotations

import math

from src.config import Settings
from src.gigachat_service import GigaChatService


class EmbeddingService:
    def __init__(self, settings: Settings):
        if settings.embedding_provider not in {"cloudru", "gigachat"}:
            raise ValueError("EMBEDDING_PROVIDER должен быть cloudru или gigachat")
        self.settings = settings

    @property
    def identity(self) -> str:
        if self.settings.embedding_provider == "cloudru":
            return f"cloudru:{self.settings.cloud_ru_embedding_model}"
        return f"gigachat:{self.settings.embedding_model}"

    def _cloud_ru_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self.settings.require_cloud_ru_credentials()
        from openai import OpenAI

        # OpenAI SDK обращается к Cloud.ru, НЕ к OpenAI.
        with OpenAI(
            api_key=self.settings.cloud_ru_api_key,
            base_url=self.settings.cloud_ru_base_url,
            timeout=60.0,
            max_retries=1,
        ) as client:
            response = client.embeddings.create(
                model=self.settings.cloud_ru_embedding_model,
                input=texts,
            )
        # API возвращает индекс исходного текста; порядок сети может отличаться.
        ordered = sorted(response.data, key=lambda item: item.index)
        if len(ordered) != len(texts) or [v.index for v in ordered] != list(range(len(texts))):
            raise RuntimeError("Cloud.ru вернул неполный или неверно индексированный ответ")
        vectors = [list(item.embedding) for item in ordered]
        if not vectors or any(not v for v in vectors):
            raise RuntimeError("Cloud.ru вернул пустой вектор")
        if len({len(v) for v in vectors}) != 1:
            raise RuntimeError("Cloud.ru вернул векторы разной размерности")
        if any(not all(math.isfinite(x) for x in v) for v in vectors):
            raise RuntimeError("Cloud.ru вернул некорректные численные значения")
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.settings.embedding_provider == "cloudru":
            # Используем ровно формат из примера API Cloud.ru; без неподтверждённых префиксов.
            return self._cloud_ru_embeddings(texts)
        return GigaChatService(self.settings).embeddings(texts)

    def embed_query(self, query: str) -> list[float]:
        if not query.strip():
            raise ValueError("Нельзя векторизовать пустой вопрос")
        if self.settings.embedding_provider == "cloudru":
            return self._cloud_ru_embeddings([query])[0]
        if self.settings.embedding_model == "EmbeddingsGigaR":
            query = (
                "Дан вопрос абитуриента, необходимо найти абзац текста с ответом "
                "в официальных документах Финансового университета.\n"
                "вопрос: " + query
            )
        return GigaChatService(self.settings).embeddings([query])[0]
