"""Offline tests. No requests sent to GigaChat or Cloud.ru."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pypdf import PdfWriter

from src.ocr_gigachat import (
    cache_directory, count_content_chars, get_pdf_page_texts,
    parse_response_text,
)
from src import ocr_gigachat


def test_counts_and_response():
    assert count_content_chars("А 1 B") == 3
    response = SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="stop", message=SimpleNamespace(content="Пункт 12. Поступающие представляют заявление о приёме и документы об образовании.")
    )])
    assert parse_response_text(response) == "Пункт 12. Поступающие представляют заявление о приёме и документы об образовании."


def test_truncation_fails():
    response = SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="length", message=SimpleNamespace(content="Очень длинный текст " * 10)
    )])
    with pytest.raises(RuntimeError, match="не полностью"):
        parse_response_text(response)


def make_blank_pdf(path: Path):
    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    with path.open("wb") as stream:
        writer.write(stream)


def test_off_rejects_scanned_pdf(tmp_path):
    pdf = tmp_path / "scan.pdf"
    make_blank_pdf(pdf)
    with pytest.raises(RuntimeError, match="не извлекается"):
        get_pdf_page_texts(pdf, cache_root=tmp_path / "processed", mode="off")


def test_auto_ocr_routes_and_uses_fake_backend(tmp_path, monkeypatch):
    pdf = tmp_path / "scan.pdf"
    make_blank_pdf(pdf)
    calls = []

    class FakeOCR:
        def __init__(self, source, *, cache_root):
            assert source == pdf
            self.cache_root = cache_root
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            pass
        def recognize(self, page_index):
            calls.append(page_index)
            return "Содержимое правил приёма: заявление, паспорт, документ об образовании."

    monkeypatch.setattr(ocr_gigachat, "GigaChatVisionOCR", FakeOCR)
    pages = get_pdf_page_texts(pdf, cache_root=tmp_path / "processed", mode="auto")
    assert len(pages) == 1
    assert calls == [0]
    assert "правил приёма" in pages[0]


def test_cache_directory_changes_with_pdf(tmp_path):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"v1")
    first = cache_directory(pdf, tmp_path, "GigaChat-2-Pro")
    pdf.write_bytes(b"v2")
    second = cache_directory(pdf, tmp_path, "GigaChat-2-Pro")
    assert first != second


def test_recognize_uploads_deletes_and_caches(tmp_path, monkeypatch):
    from src.ocr_gigachat import GigaChatVisionOCR

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"PDF fake for testing cache")
    log = []

    class FakeClient:
        def upload_file(self, file, purpose):
            log.append(("upload", purpose))
            assert file.read(4) == b"jpeg"
            return SimpleNamespace(id_="uploaded-file-id")

        def chat(self, request):
            assert request["messages"][0]["attachments"] == ["uploaded-file-id"]
            log.append(("chat", request["model"]))
            return SimpleNamespace(choices=[SimpleNamespace(
                finish_reason="stop", message=SimpleNamespace(content="Достоверный распознанный текст, содержащий много кириллических символов и номер пункта 15.")
            )])

        def delete_file(self, file_id):
            log.append(("delete", file_id))

    def fake_render(pdf_path, page_index, destination):
        destination.write_bytes(b"jpeg mocked bytes")

    monkeypatch.setattr(ocr_gigachat, "render_page_to_jpeg", fake_render)
    instance = GigaChatVisionOCR.__new__(GigaChatVisionOCR)
    instance.pdf_path = pdf
    instance.model = "GigaChat-2-Pro"
    instance.cache_dir = tmp_path / "cached"
    instance.cache_dir.mkdir()
    instance._client = FakeClient()
    text1 = instance.recognize(0)
    text2 = instance.recognize(0)
    assert text1 == text2
    assert log == [
        ("upload", "general"),
        ("chat", "GigaChat-2-Pro"),
        ("delete", "uploaded-file-id"),
    ]


def test_upload_with_legacy_id_field(tmp_path, monkeypatch):
    """SDK versions exposing `id` must still work without network access."""
    from src.ocr_gigachat import GigaChatVisionOCR
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"fake pdf")
    class FakeClient:
        def upload_file(self, file, purpose):
            return SimpleNamespace(id="legacy-id")
        def chat(self, request):
            assert request["messages"][0]["attachments"] == ["legacy-id"]
            return SimpleNamespace(choices=[SimpleNamespace(
                finish_reason="stop", message=SimpleNamespace(content="Это распознанный текст страницы нормативного документа, не менее двадцати пяти символов.")
            )])
        def delete_file(self, file_id):
            assert file_id == "legacy-id"
    monkeypatch.setattr(ocr_gigachat, "render_page_to_jpeg", lambda pdf_path, page_index, out_path: out_path.write_bytes(b"jpeg"))
    obj = GigaChatVisionOCR.__new__(GigaChatVisionOCR)
    obj.pdf_path = pdf
    obj.model = "GigaChat-2-Pro"
    obj.cache_dir = tmp_path / "ocr-cache"
    obj.cache_dir.mkdir()
    obj._client = FakeClient()
    assert "нормативного" in obj.recognize(0)


def test_moderation_is_not_ocr(monkeypatch, tmp_path):
    from src.ocr_gigachat import is_moderation_refusal, GigaChatVisionOCR

    refusal = (
        "К сожалению, иногда генеративные языковые модели могут создавать "
        "некорректные ответы, основанные на открытых источниках. Во избежание "
        "неправильного толкования, ответы на вопросы, связанные с чувствительными "
        "темами, временно ограничены. Благодарим за понимание."
    )
    assert is_moderation_refusal(refusal)
    assert not is_moderation_refusal(
        "В соответствии с правилами приема Правительства Российской Федерации "
        "предоставляются особые права."
    )
    for reason, content in [("blacklist", "Пункт правил приема о документах." * 4),
                            ("stop", refusal)]:
        response = SimpleNamespace(choices=[SimpleNamespace(
            finish_reason=reason, message=SimpleNamespace(content=content)
        )])
        with pytest.raises(RuntimeError, match="тематическим ограничениям|модерационного отказа"):
            parse_response_text(response)

    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "page_0013.txt").write_text(refusal, encoding="utf-8")
    obj = GigaChatVisionOCR.__new__(GigaChatVisionOCR)
    obj.cache_dir = cache
    obj._client = None
    assert obj.recognize(12) == ""
    assert not (cache / "page_0013.txt").exists()
    assert (cache / "page_0013.skipped").exists()


def test_provider_refusal_skips_page_without_cache_and_without_second_api_call(tmp_path, monkeypatch):
    from src.ocr_gigachat import GigaChatVisionOCR

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"fake pdf")
    calls = []

    class FakeClient:
        def upload_file(self, file, purpose):
            calls.append("upload")
            return SimpleNamespace(id_="file-id")
        def chat(self, request):
            calls.append("chat")
            return SimpleNamespace(choices=[SimpleNamespace(
                finish_reason="blacklist", message=SimpleNamespace(content="")
            )])
        def delete_file(self, file_id):
            calls.append("delete")

    monkeypatch.setattr(ocr_gigachat, "render_page_to_jpeg", lambda pdf_path, page_index, out_path: out_path.write_bytes(b"jpeg"))
    obj = GigaChatVisionOCR.__new__(GigaChatVisionOCR)
    obj.pdf_path = pdf
    obj.model = "GigaChat-2-Pro"
    obj.cache_dir = tmp_path / "ocr-cache"
    obj.cache_dir.mkdir()
    obj._client = FakeClient()

    assert obj.recognize(0) == ""
    assert not (obj.cache_dir / "page_0001.txt").exists()
    assert (obj.cache_dir / "page_0001.skipped").exists()
    assert obj.recognize(0) == ""
    assert calls == ["upload", "chat", "delete"]


def test_other_api_errors_are_not_swallowed(tmp_path, monkeypatch):
    from src.ocr_gigachat import GigaChatVisionOCR
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"fake pdf")
    class FakeClient:
        def upload_file(self, file, purpose):
            return SimpleNamespace(id_="file-id")
        def chat(self, request):
            raise ConnectionError("connection failed")
        def delete_file(self, file_id):
            pass
    monkeypatch.setattr(ocr_gigachat, "render_page_to_jpeg", lambda pdf_path, page_index, out_path: out_path.write_bytes(b"jpeg"))
    obj = GigaChatVisionOCR.__new__(GigaChatVisionOCR)
    obj.pdf_path = pdf
    obj.model = "GigaChat-2-Pro"
    obj.cache_dir = tmp_path / "ocr-cache"
    obj.cache_dir.mkdir()
    obj._client = FakeClient()
    with pytest.raises(ConnectionError, match="connection failed"):
        obj.recognize(0)
    assert not (obj.cache_dir / "page_0001.skipped").exists()


def test_multiple_pages_include_empty_refusal(tmp_path, monkeypatch):
    pdf = tmp_path / "scan.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    writer.add_blank_page(width=595, height=842)
    with pdf.open("wb") as file:
        writer.write(file)
    calls = []
    class FakeOCR:
        def __init__(self, source, *, cache_root):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def recognize(self, idx):
            calls.append(idx)
            return "" if idx == 0 else "Достоверный текст второго листа документа."
    monkeypatch.setattr(ocr_gigachat, "GigaChatVisionOCR", FakeOCR)
    texts = get_pdf_page_texts(pdf, cache_root=tmp_path / "cache", mode="auto")
    assert texts == ["", "Достоверный текст второго листа документа."]
    assert calls == [0, 1]
