"""Turn user files into plain text."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = (".txt", ".md", ".markdown", ".docx", ".pdf")


class IngestError(Exception):
    """A file could not be read as text."""


@dataclass
class Document:
    path: Path
    text: str

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def read_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise IngestError(f"Unsupported file type '{suffix or path.name}'. Use: {', '.join(SUPPORTED_EXTENSIONS)}")
    try:
        if suffix == ".docx":
            return _read_docx(path)
        if suffix == ".pdf":
            return _read_pdf(path)
        return _read_plain(path)
    except IngestError:
        raise
    except Exception as exc:  # parsing libraries raise many different types
        raise IngestError(f"Could not read {path.name}: {exc}") from exc


def load_document(path: str | Path) -> Document:
    p = Path(path)
    if not p.is_file():
        raise IngestError(f"Not a file: {p}")
    text = read_text(p).strip()
    if not text:
        raise IngestError(f"{p.name} contains no readable text")
    return Document(path=p, text=text)


def _read_plain(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _read_docx(path: Path) -> str:
    import docx

    document = docx.Document(str(path))
    return "\n\n".join(p.text for p in document.paragraphs if p.text.strip())


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(p for p in pages if p.strip())
    if not text.strip():
        raise IngestError(f"{path.name} has no extractable text (it may be a scanned PDF)")
    return text
