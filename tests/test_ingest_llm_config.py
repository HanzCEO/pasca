from pathlib import Path

import pytest

from pasca import config, llm
from pasca.ingest import Document, IngestError, load_document


def test_load_plain_text(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_text("Hello there. General Kenobi.", encoding="utf-8")
    doc = load_document(f)
    assert doc.word_count == 4


def test_load_docx(tmp_path: Path):
    import docx

    d = docx.Document()
    d.add_paragraph("First paragraph.")
    d.add_paragraph("Second one.")
    f = tmp_path / "a.docx"
    d.save(f)
    assert load_document(f).text == "First paragraph.\n\nSecond one."


def test_rejects_unsupported_and_empty(tmp_path: Path):
    bad = tmp_path / "a.exe"
    bad.write_bytes(b"x")
    with pytest.raises(IngestError):
        load_document(bad)
    empty = tmp_path / "e.txt"
    empty.write_text("   ")
    with pytest.raises(IngestError):
        load_document(empty)


def test_excerpt_budget_redistributes():
    short = Document(Path("s.txt"), "short text")
    long = Document(Path("l.txt"), "word " * 1000)
    excerpts, truncated = llm.build_excerpts([short, long], budget=1000)
    assert truncated
    assert "short text" in excerpts
    assert excerpts.count("word") > 150  # long doc got the short doc's unused share


def test_excerpt_no_truncation_when_fits():
    _, truncated = llm.build_excerpts([Document(Path("a.txt"), "tiny")], budget=1000)
    assert not truncated


def test_prepare_includes_metrics():
    req = llm.prepare([Document(Path("a.txt"), "Some text.")], {"size": {"words": 2}}, config.Settings(model="m"))
    assert req.messages[0]["role"] == "system"
    assert '"words": 2' in req.messages[1]["content"]


def test_config_roundtrip_hides_key(tmp_path: Path):
    path = tmp_path / "c.toml"
    s = config.Settings(base_url="http://x/v1", model='m "q"', api_key="secret", temperature=0.5)
    config.save(s, path)
    assert "secret" not in path.read_text()
    loaded = config.load(path)
    assert loaded.model == 'm "q"' and loaded.temperature == 0.5 and loaded.api_key == ""

    s.save_api_key = True
    config.save(s, path)
    assert config.load(path).api_key == "secret"
