from src.chunking import split_text


def test_empty():
    assert split_text("  ") == []


def test_chunk_coverage():
    chunks = split_text("Начало. " * 450, chunk_size=350, overlap=50)
    assert len(chunks) > 4
    assert all(chunks)
    assert "Начало" in chunks[0] and "Начало" in chunks[-1]


def test_invalid_overlap():
    import pytest
    with pytest.raises(ValueError):
        split_text("Текст", chunk_size=300, overlap=151)
