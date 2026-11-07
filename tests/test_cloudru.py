from dataclasses import replace
from types import SimpleNamespace
import sys

import pytest

from src.config import get_settings
from src.embeddings import EmbeddingService


def test_cloudru_defaults(monkeypatch):
    for key in ("EMBEDDING_PROVIDER", "CLOUD_RU_EMBEDDING_MODEL", "CLOUD_RU_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    settings = get_settings()
    assert settings.embedding_provider == "cloudru"
    assert settings.cloud_ru_embedding_model == "ai-sage/Giga-Embeddings-instruct-3B"
    assert settings.cloud_ru_base_url == "https://foundation-models.api.cloud.ru/v1"


def test_cloudru_missing_key():
    settings = replace(get_settings(), embedding_provider="cloudru", cloud_ru_api_key="")
    with pytest.raises(ValueError, match="CLOUD_RU_API_KEY"):
        EmbeddingService(settings).embed_query("Какие ЕГЭ нужны?")


def test_cloudru_batch_mock(monkeypatch):
    settings = replace(get_settings(), embedding_provider="cloudru", cloud_ru_api_key="test-only")
    calls = []

    class FakeEmbeddings:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(data=[
                SimpleNamespace(index=1, embedding=[0.0, 1.0]),
                SimpleNamespace(index=0, embedding=[1.0, 0.0]),
            ])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            assert kwargs["base_url"] == "https://foundation-models.api.cloud.ru/v1"
            assert kwargs["api_key"] == "test-only"
            self.embeddings = FakeEmbeddings()
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    result = EmbeddingService(settings).embed_documents(["Текст 1", "Текст 2"])
    assert result == [[1.0, 0.0], [0.0, 1.0]]
    assert calls[0]["model"] == "ai-sage/Giga-Embeddings-instruct-3B"
    assert calls[0]["input"] == ["Текст 1", "Текст 2"]


def test_old_provider_still_available(monkeypatch):
    settings = replace(get_settings(), embedding_provider="gigachat")
    assert EmbeddingService(settings).identity == "gigachat:EmbeddingsGigaR"


def test_reject_unknown_provider():
    settings = replace(get_settings(), embedding_provider="openrouter")
    with pytest.raises(ValueError, match="EMBEDDING_PROVIDER"):
        EmbeddingService(settings)
