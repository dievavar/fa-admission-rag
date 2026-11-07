"""Создание индекса. Пример: python -m src.ingest --reset"""
from __future__ import annotations

import argparse

from src.chunking import split_text
from src.config import get_settings
from src.documents import load_pages
from src.embeddings import EmbeddingService
from src.vector_store import make_client, rebuild_collection, upload_chunks


def make_chunks(pages: list, *, size: int = 1200, overlap: int = 160) -> list[dict]:
    chunks = []
    for page in pages:
        for index, text in enumerate(split_text(page.text, size, overlap)):
            chunks.append({
                "text": text,
                "filename": page.filename,
                "title": page.title,
                "source_url": page.source_url,
                "campaign_year": page.campaign_year,
                "page": page.page,
                "chunk_index": index,
            })
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Индексирует файлы из data/raw/ в Qdrant")
    parser.add_argument("--reset", action="store_true", help="Удалить прежнюю коллекцию и создать новую")
    args = parser.parse_args()
    settings = get_settings()
    pages = [page for page in load_pages(settings.data_dir) if page.campaign_year == settings.admission_year]
    if not pages:
        raise SystemExit(f"В data/raw нет документов для кампании {settings.admission_year} года")
    chunks = make_chunks(pages)
    embeddings = EmbeddingService(settings)
    print(f"Источников: {len({p.filename for p in pages})}, фрагментов: {len(chunks)}")
    vectors = []
    for start in range(0, len(chunks), 8):
        batch = [item["text"] for item in chunks[start:start + 8]]
        vectors.extend(embeddings.embed_documents(batch))
        print(f"Векторизовано {len(vectors)}/{len(chunks)}")
    client = make_client(settings)
    try:
        rebuild_collection(client, settings, len(vectors[0]), reset=args.reset)
        upload_chunks(client, settings, chunks, vectors, embeddings.identity)
    finally:
        client.close()
    print(f"Готово. Коллекция {settings.collection} ({embeddings.identity})")


if __name__ == "__main__":
    main()
