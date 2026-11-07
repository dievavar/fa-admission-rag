"""Retrieval + generation с обязательными ссылками на документы."""
from __future__ import annotations

from dataclasses import dataclass

from src.config import Settings, get_settings
from src.embeddings import EmbeddingService
from src.gigachat_service import GigaChatService
from src.vector_store import SearchHit, make_client, search


NO_DATA = "В загруженных документах нет подтверждённой информации по этому вопросу."


@dataclass(frozen=True)
class RAGAnswer:
    text: str
    sources: list[SearchHit]


def ask(question: str, settings: Settings | None = None) -> RAGAnswer:
    settings = settings or get_settings()
    question = question.strip()
    if not question:
        return RAGAnswer("Введите вопрос.", [])
    if len(question) > 2000:
        return RAGAnswer("Вопрос слишком длинный (максимум 2000 символов).", [])

    embeddings = EmbeddingService(settings)
    vector = embeddings.embed_query(question)
    client = make_client(settings)
    try:
        hits = search(client, settings, vector, embeddings.identity)
    finally:
        client.close()

    if not hits:
        return RAGAnswer(NO_DATA, [])
    lines = []
    for i, hit in enumerate(hits, start=1):
        page = f", стр. {hit.page}" if hit.page else ""
        lines.append(
            f"[{i}] {hit.title}{page}; приёмная кампания {hit.campaign_year}; "
            f"источник: {hit.source_url}\n{hit.text}"
        )
    context = "\n\n".join(lines)
    text = GigaChatService(settings).answer(question, context)
    return RAGAnswer(text or NO_DATA, hits)
