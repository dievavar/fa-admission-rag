"""Эмбеддинги через официальный GigaChat API; локальных моделей нет."""
from __future__ import annotations

from src.config import Settings
from src.gigachat_service import GigaChatService


class EmbeddingService:
    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def identity(self) -> str:
        return f"gigachat:{self.settings.embedding_model}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Для EmbeddingsGigaR документы кодируем без поисковой инструкции."""
        if not texts:
            return []
        return GigaChatService(self.settings).embeddings(texts)

    def embed_query(self, query: str) -> list[float]:
        """Для EmbeddingsGigaR добавляем retrieval-инструкцию к вопросу."""
        if self.settings.embedding_model == "EmbeddingsGigaR":
            query = (
                "Дан вопрос абитуриента, необходимо найти абзац текста с ответом "
                "в официальных документах Финансового университета.\n"
                "вопрос: " + query
            )
        return GigaChatService(self.settings).embeddings([query])[0]
