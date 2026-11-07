"""Qdrant через HTTP: Docker локально или облачный сервис."""
from __future__ import annotations

from dataclasses import dataclass
from uuid import NAMESPACE_URL, uuid5

from src.config import Settings


@dataclass(frozen=True)
class SearchHit:
    text: str
    filename: str
    title: str
    source_url: str
    campaign_year: int
    page: int | None
    score: float


def make_client(settings: Settings):
    from qdrant_client import QdrantClient

    if not settings.qdrant_url:
        raise ValueError("Не задан QDRANT_URL. Для Docker: http://localhost:6333")
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        timeout=30,
    )


def rebuild_collection(client, settings: Settings, dimension: int, reset: bool) -> None:
    from qdrant_client import models

    exists = client.collection_exists(settings.collection)
    if exists and reset:
        client.delete_collection(settings.collection)
        exists = False
    if exists:
        actual = client.get_collection(settings.collection).config.params.vectors
        # В этом MVP используется один безымянный вектор.
        if getattr(actual, "size", None) != dimension:
            raise ValueError(
                "Размерность эмбеддингов не совпадает с существующей коллекцией. "
                "Запустите python -m src.ingest --reset после смены модели."
            )
    else:
        client.create_collection(
            collection_name=settings.collection,
            vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
        )


def upload_chunks(client, settings: Settings, chunks: list[dict], vectors: list[list[float]], embedding_identity: str) -> None:
    from qdrant_client import models

    if len(chunks) != len(vectors):
        raise ValueError("Число чанков и эмбеддингов не совпадает")
    points = []
    for data, vector in zip(chunks, vectors):
        stable_key = f"{data['filename']}:{data['page']}:{data['chunk_index']}"
        points.append(
            models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, stable_key)),
                vector=vector,
                payload={**data, "embedding_model": embedding_identity},
            )
        )
    for start in range(0, len(points), 50):
        client.upsert(collection_name=settings.collection, points=points[start : start + 50], wait=True)


def search(client, settings: Settings, vector: list[float], embedding_identity: str) -> list[SearchHit]:
    if not client.collection_exists(settings.collection):
        raise ValueError("Коллекция ещё не создана. Сначала запустите python -m src.ingest --reset")
    from qdrant_client import models

    result = client.query_points(
        collection_name=settings.collection,
        query=vector,
        limit=settings.top_k,
        with_payload=True,
        query_filter=models.Filter(must=[models.FieldCondition(key="campaign_year", match=models.MatchValue(value=settings.admission_year))]),
    ).points
    hits = []
    for item in result:
        payload = item.payload or {}
        if payload.get("embedding_model") != embedding_identity:
            raise ValueError("Коллекция проиндексирована другой моделью. Повторите ingest --reset")
        if float(item.score) < settings.min_score:
            continue
        hits.append(
            SearchHit(
                text=str(payload.get("text", "")),
                filename=str(payload.get("filename", "")),
                title=str(payload.get("title", "")),
                source_url=str(payload.get("source_url", "")),
                campaign_year=int(payload.get("campaign_year", 0)),
                page=payload.get("page"),
                score=float(item.score),
            )
        )
    return hits
