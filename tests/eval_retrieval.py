"""Оценка поиска RAG без генерации ответов GigaChat.

Запуск из корня проекта:
    python tests/eval_retrieval.py --mode offline
    python tests/eval_retrieval.py --mode live

В offline использует сохранённые поисковые результаты без API.
В live использует реальный src.embeddings + src.vector_store; API эмбеддингов
может быть платным. Векторы запросов кешируются для повторных запусков.
"""
from __future__ import annotations

import argparse
from dataclasses import is_dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CASES = Path(__file__).with_name("retrieval_cases.json")
DEFAULT_HITS = Path(__file__).with_name("retrieval_offline_hits.json")
DEFAULT_REPORT = ROOT / "data" / "processed" / "eval" / "retrieval_report.json"
DEFAULT_CACHE = ROOT / "data" / "processed" / "eval" / "query_vectors.json"


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Файл не найден: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Ошибка JSON в {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Ожидался JSON-объект: {path}")
    return data


def key(filename: str, page: int | None) -> tuple[str, int | None]:
    return Path(filename).name.casefold(), page


def evaluate_case(case: dict[str, Any], hits: list[dict[str, Any]], k: int) -> dict[str, Any]:
    """Hit@K, page Recall@K, RR@K по уникальным (файл, страница)."""
    gold = {
        key(str(ref["filename"]), int(ref["page"]))
        for ref in case["relevant_locations"]
    }
    if not gold:
        raise ValueError(f"Нет эталонных страниц для {case['id']}")

    top = hits[:k]
    found = {key(str(hit.get("filename", "")), hit.get("page")) for hit in top}
    matched = gold & found
    first = next(
        (position for position, hit in enumerate(top, start=1)
         if key(str(hit.get("filename", "")), hit.get("page")) in gold),
        None,
    )
    return {
        "id": case["id"],
        "question": case["question"],
        "verified_gold": bool(case.get("verified", False)),
        "hit_at_k": int(bool(matched)),
        "page_recall_at_k": len(matched) / len(gold),
        "reciprocal_rank_at_k": 1 / first if first is not None else 0.0,
        "first_relevant_rank": first,
        "expected_pages": sorted({p for _, p in gold}),
        "retrieved_pages": [hit.get("page") for hit in top],
        "retrieved": top,
        "matched_locations": [
            {"filename": filename, "page": page}
            for filename, page in sorted(matched)
        ],
    }


def offline_hits(path: Path) -> dict[str, list[dict[str, Any]]]:
    data = load_json(path)
    result = {}
    for item in data.get("queries", []):
        result[item["id"]] = item["hits"]
    return result


def update_top_k(settings: Any, k: int | None) -> Any:
    if k is None or k == settings.top_k:
        return settings
    if is_dataclass(settings):
        return replace(settings, top_k=k)
    if callable(getattr(settings, "model_copy", None)):
        return settings.model_copy(update={"top_k": k})
    raise SystemExit("Невозможно переопределить TOP_K для текущего типа Settings; запусти без --k.")


def live_hits(
    cases: list[dict[str, Any]],
    cache_path: Path,
    k: int | None,
) -> tuple[dict[str, list[dict[str, Any]]], int, int, int]:
    # Импорт в live-ветке: offline запускается без Qdrant/Cloud.ru.
    from src.config import get_settings
    from src.embeddings import EmbeddingService
    from src.vector_store import make_client, search

    settings = update_top_k(get_settings(), k)
    embeddings = EmbeddingService(settings)
    identity = str(embeddings.identity)
    cache = load_json(cache_path) if cache_path.is_file() else {}
    if cache.get("embedding_identity") != identity:
        cache = {"embedding_identity": identity, "vectors": {}}
    cached_vectors = cache.get("vectors", {})
    if not isinstance(cached_vectors, dict):
        cached_vectors = {}
    cache["vectors"] = cached_vectors

    result: dict[str, list[dict[str, Any]]] = {}
    api_calls = 0
    cache_hits = 0
    client = make_client(settings)
    try:
        for index, case in enumerate(cases, start=1):
            question = case["question"].strip()
            digest = sha256(question.encode("utf-8")).hexdigest()
            vector = cached_vectors.get(digest)
            if vector is None:
                vector = embeddings.embed_query(question)
                cached_vectors[digest] = vector
                api_calls += 1
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            else:
                cache_hits += 1
            hits = search(client, settings, vector, embeddings.identity)
            result[case["id"]] = [
                {
                    "filename": hit.filename,
                    "page": hit.page,
                    "score": round(hit.score, 6),
                    "title": hit.title,
                }
                for hit in hits
            ]
            print(f"  [{index}/{len(cases)}] {case['id']}: {len(hits)} фрагм.")
    finally:
        client.close()
    return result, int(settings.top_k), api_calls, cache_hits


def main() -> None:
    parser = argparse.ArgumentParser(description="Оценка retrieval для RAG Финансового университета")
    parser.add_argument("--mode", choices=["offline", "live"], default="offline")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--offline-hits", type=Path, default=DEFAULT_HITS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--k", type=int, default=None, help="По умолчанию: 4 (offline) или settings.top_k (live)")
    args = parser.parse_args()
    if args.k is not None and args.k < 1:
        parser.error("--k должен быть положительным целым числом")

    cases = load_json(args.cases).get("cases", [])
    if not cases:
        raise SystemExit("Список cases пуст")
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise SystemExit("ID вопросов должны быть уникальными")

    if args.mode == "offline":
        by_id = offline_hits(args.offline_hits)
        k = args.k or 4
        api_calls = 0
        cache_hits = 0
        print("OFFLINE: используем сохранённую выдачу, API не вызываем.")
    else:
        print("LIVE: получение новых эмбеддингов может тарифицироваться Cloud.ru.")
        by_id, k, api_calls, cache_hits = live_hits(cases, args.cache, args.k)

    missing = [case["id"] for case in cases if case["id"] not in by_id]
    if missing:
        raise SystemExit("Нет выдачи для кейсов: " + ", ".join(missing))

    rows = [evaluate_case(case, by_id[case["id"]], k) for case in cases]
    count = len(rows)
    metrics = {
        "hit_at_k": sum(row["hit_at_k"] for row in rows) / count,
        "mean_page_recall_at_k": sum(row["page_recall_at_k"] for row in rows) / count,
        "mrr_at_k": sum(row["reciprocal_rank_at_k"] for row in rows) / count,
    }

    print(f"\n{'Кейс':<23} {'Топ страниц':<21} {'Ранг':>5} {'Hit':>5} {'Recall':>7} {'RR':>7}")
    print("-" * 78)
    for row in rows:
        pages = ",".join("?" if p is None else str(p) for p in row["retrieved_pages"])
        print(
            f"{row['id']:<23} {pages[:21]:<21} "
            f"{str(row['first_relevant_rank'] or '-'):>5} "
            f"{row['hit_at_k']:>5} {row['page_recall_at_k']:>7.3f} "
            f"{row['reciprocal_rank_at_k']:>7.3f}"
        )
    print(f"\nHit@{k}:         {metrics['hit_at_k']:.1%}")
    print(f"Page Recall@{k}: {metrics['mean_page_recall_at_k']:.1%}")
    print(f"MRR@{k}:         {metrics['mrr_at_k']:.3f}")
    if any(not row["verified_gold"] for row in rows):
        print("\nВАЖНО: эталонные страницы предварительные (verified=false).")
        print("Сверь их по исходному PDF перед использованием метрик в курсовой.")

    report = {
        "mode": args.mode,
        "k": k,
        "case_count": count,
        "api_embedding_calls": api_calls,
        "cached_embedding_queries": cache_hits,
        "gold_verified": all(row["verified_gold"] for row in rows),
        "metrics": metrics,
        "cases": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {args.report}")


if __name__ == "__main__":
    main()
