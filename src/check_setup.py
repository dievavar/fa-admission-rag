"""Проверка Qdrant и API эмбеддингов (ровно один тестовый запрос)."""
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

    service = EmbeddingService(settings)
    print(f"Проверка эмбеддингов: {service.identity}")
    vector = service.embed_query("Как поступить в Финансовый университет?")
    print(f"Эмбеддинги получены: размерность {len(vector)}")
    print("Готово. Следующий шаг: python -m src.ingest --reset")


if __name__ == "__main__":
    main()
