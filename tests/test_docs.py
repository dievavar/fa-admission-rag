from pathlib import Path

from src.documents import load_pages
from src.ingest import make_chunks


def test_sample_files_and_metadata():
    root = Path(__file__).resolve().parents[1]
    pages = load_pages(root / "data" / "raw")
    assert len(pages) == 4
    assert all(page.campaign_year == 2026 and page.source_url.startswith("https://www.fa.ru/") for page in pages)
    assert all(page.text for page in pages)
    chunks = make_chunks(pages)
    assert chunks and all(chunk["filename"] for chunk in chunks)
