"""Прозрачное разбиение текста с небольшим перекрытием фрагментов."""
from __future__ import annotations


def split_text(text: str, chunk_size: int = 1200, overlap: int = 160) -> list[str]:
    if chunk_size < 100 or overlap < 0 or overlap >= chunk_size // 2:
        raise ValueError("Размер чанка >= 100, перекрытие < половины чанка.")
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        if end < len(text):
            # Ищем границу в последней трети окна, предпочтительно конец абзаца.
            min_boundary = start + int(chunk_size * 0.65)
            paragraph = text.rfind("\n\n", min_boundary, end)
            whitespace = text.rfind(" ", min_boundary, end)
            if paragraph != -1:
                end = paragraph + 2
            elif whitespace != -1:
                end = whitespace + 1
        part = text[start:end].strip()
        if part:
            chunks.append(part)
        if end == len(text):
            break
        # Следующий чанк немного повторяет предыдущий; не зацикливаемся.
        start = max(start + 1, end - overlap)
    return chunks
