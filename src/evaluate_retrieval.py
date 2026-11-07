"""Простой Hit@k тест качества поиска на размеченных вопросах.

Запуск после индексации: python -m src.evaluate_retrieval
Сменив модель, перестройте индекс и сравните результаты на том же наборе.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.config import get_settings, ROOT
from src.embeddings import EmbeddingService
from src.vector_store import make_client, search


def main() -> None:
    settings = get_settings()
    examples = [json.loads(line) for line in (ROOT / "tests" / "questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    labeled = [item for item in examples if "expected_source" in item]
    if not labeled:
        raise SystemExit("Не размечены правильные источники (expected_source).")
    embedding = EmbeddingService(settings)
    client = make_client(settings)
    hits_count = 0
    try:
        for item in labeled:
            vector = embedding.embed_query(item["question"])
            results = search(client, settings, vector, embedding.identity)
            names = [h.filename for h in results]
            ok = item["expected_source"] in names
            hits_count += int(ok)
            print(f"{'OK' if ok else 'MISS'} {item['question']}")
            print(f"  ожидали: {item['expected_source']}; получили: {names}")
    finally:
        client.close()
    print(f"Hit@{settings.top_k}: {hits_count}/{len(labeled)} ({hits_count/len(labeled):.0%})")
    print("Это маленькая пилотная выборка, а не объективный бенчмарк. Для курсовой нужно 40–60 вопросов.")


if __name__ == "__main__":
    main()
