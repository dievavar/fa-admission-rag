"""Test metadata and chunking using an isolated small fixture (no external API)."""
import json

from src.documents import load_pages
from src.ingest import make_chunks


def test_sample_files_and_metadata(tmp_path):
    # Don't read data/raw from the user's workspace: it may contain scanned
    # PDFs that trigger real (possibly billable) requests to GigaChat.
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "rules_demo.md").write_text(
        "Правила приёма. Заявления на поступление принимаются согласно "
        "официальным условиям университета.",
        encoding="utf-8",
    )
    (raw / "manifest.json").write_text(
        json.dumps({
            "rules_demo.md": {
                "title": "Тестовый документ о приёме",
                "source_url": "https://www.fa.ru/for-applicants/priyemka/dokumenti/",
                "campaign_year": 2026,
            }
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    pages = load_pages(raw)
    assert len(pages) == 1
    assert pages[0].campaign_year == 2026
    assert pages[0].source_url.startswith("https://www.fa.ru/")
    assert pages[0].text
    chunks = make_chunks(pages)
    assert chunks
    assert all(chunk["filename"] == "rules_demo.md" for chunk in chunks)
