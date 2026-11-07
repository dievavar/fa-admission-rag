from src.config import get_settings


def test_default_is_api_and_docker(monkeypatch):
    for name in ("QDRANT_URL", "GIGACHAT_EMBEDDING_MODEL", "GIGACHAT_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    settings = get_settings()
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.embedding_model == "EmbeddingsGigaR"
    assert settings.base_url == "https://api.giga.chat/v1"


def test_qdrant_api_key_env(monkeypatch):
    monkeypatch.setenv("QDRANT_API_KEY", "secret-example")
    assert get_settings().qdrant_api_key == "secret-example"
