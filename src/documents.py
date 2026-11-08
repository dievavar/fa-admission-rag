"""Чтение PDF/TXT/MD с указанием источника, страницы и года кампании."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from src.ocr_gigachat import get_pdf_page_texts


@dataclass(frozen=True)
class SourcePage:
    text: str
    filename: str
    title: str
    source_url: str
    campaign_year: int
    page: int | None


def clean_text(value: str) -> str:
    value = value.replace("\u00a0", " ").replace("\r\n", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def read_manifest(data_dir: Path) -> dict:
    path = data_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Нет реестра источников: {path}")
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json должен содержать объект с именами файлов")
    return manifest


def load_pages(data_dir: Path) -> list[SourcePage]:
    manifest = read_manifest(data_dir)
    result: list[SourcePage] = []
    valid_types = {".pdf", ".txt", ".md"}
    for file_path in sorted(data_dir.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() not in valid_types:
            continue
        if file_path.name not in manifest:
            raise ValueError(
                f"Для {file_path.name} нет записи в data/raw/manifest.json"
            )
        meta = manifest[file_path.name]
        required = ("title", "source_url", "campaign_year")
        if any(key not in meta for key in required):
            raise ValueError(f"Для {file_path.name} обязательны поля {required}")
        year = int(meta["campaign_year"])
        url = str(meta["source_url"])
        title = str(meta["title"])

        if file_path.suffix.lower() == ".pdf":
            page_texts = get_pdf_page_texts(
                file_path,
                cache_root=data_dir.parent / "processed" / "ocr_gigachat",
                mode=str(meta.get("ocr", "auto")).lower(),
            )
            for page_no, raw_text in enumerate(page_texts, start=1):
                text = clean_text(raw_text)
                if text:
                    result.append(SourcePage(
                        text, file_path.name, title, url, year, page_no
                    ))
        else:
            content = clean_text(file_path.read_text(encoding="utf-8"))
            if content:
                result.append(SourcePage(
                    content, file_path.name, title, url, year, None
                ))
    return result
