"""Быстрая проверка Docker Qdrant и доступа к GigaChat API.

Запуск: python -m src.check_setup
Делает один платный/квотируемый запрос эмбеддинга.
"""
from __future__ import annotations

from src.config import get_settings
from src.embeddings import EmbeddingService
from src.vector_store import make_client


def main() -> None:
    settings = get_settings()
    print(f"Qdrant: {settings.qdrant_url}")
    client = make_client(settings)
    try:
        collections = client.get_collections()
        print(f"Qdrant работает, коллекций: {len(collections.collections)}")
    finally:
        client.close()

    settings.require_credentials()
    result = EmbeddingService(settings).embed_query("Как поступить в Финансовый университет?")
    if not result:
        raise RuntimeError("Пустой эмбеддинг из API")
    print(f"GigaChat API работает: {settings.embedding_model}, размерность {len(result)}")
    print("Проверка пройдена. Теперь: python -m src.ingest --reset")


if __name__ == "__main__":
    main()
