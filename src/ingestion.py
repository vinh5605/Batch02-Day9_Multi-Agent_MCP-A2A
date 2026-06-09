"""Document ingestion pipeline.

Supports:
  - Uploaded files: PDF, DOCX, MD, TXT, JSON, HTML
  - Web URLs (via urllib)

All ingested content is:
  1. Saved raw to data/landing/uploads/
  2. Converted to markdown and saved to data/standardized/uploads/
  3. Chunked and appended to data/local_chunks.json (rebuilt on refresh)
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[1]
_LANDING = _ROOT / "data" / "landing" / "uploads"
_STANDARD = _ROOT / "data" / "standardized" / "uploads"
_CHUNKS_PATH = _ROOT / "data" / "local_chunks.json"
_CHUNK_SIZE = 400  # words per chunk


def _ensure_dirs() -> None:
    _LANDING.mkdir(parents=True, exist_ok=True)
    _STANDARD.mkdir(parents=True, exist_ok=True)
    (_ROOT / "data").mkdir(parents=True, exist_ok=True)


def _load_chunks() -> list[dict]:
    if _CHUNKS_PATH.exists():
        return json.loads(_CHUNKS_PATH.read_text(encoding="utf-8"))
    return []


def _save_chunks(chunks: list[dict]) -> None:
    _CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")


def _to_markdown(filename: str, content: bytes) -> str:
    """Convert file bytes to plain markdown text."""
    ext = Path(filename).suffix.lower()
    try:
        if ext in (".md", ".txt"):
            return content.decode("utf-8", errors="replace")
        if ext == ".json":
            data = json.loads(content.decode("utf-8", errors="replace"))
            if isinstance(data, list):
                return "\n\n".join(str(item) for item in data)
            return json.dumps(data, ensure_ascii=False, indent=2)
        if ext in (".html", ".htm"):
            text = content.decode("utf-8", errors="replace")
            # strip HTML tags
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"&[a-z]+;", " ", text)
            return re.sub(r"\s+", " ", text).strip()
        if ext == ".pdf":
            try:
                import io
                try:
                    import pypdf as _pdf_lib
                    reader = _pdf_lib.PdfReader(io.BytesIO(content))
                    return "\n\n".join(p.extract_text() or "" for p in reader.pages)
                except ImportError:
                    pass
                try:
                    import PyPDF2 as _pdf_lib2  # noqa: N813
                    reader2 = _pdf_lib2.PdfReader(io.BytesIO(content))
                    return "\n\n".join(
                        (reader2.pages[i].extract_text() or "") for i in range(len(reader2.pages))
                    )
                except ImportError:
                    pass
            except Exception:
                pass
            return content.decode("utf-8", errors="replace")
        if ext in (".docx", ".doc"):
            try:
                import io
                import docx
                doc = docx.Document(io.BytesIO(content))
                return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
            except Exception:
                pass
        return content.decode("utf-8", errors="replace")
    except Exception:
        return content.decode("utf-8", errors="replace")


def _chunk_text(text: str, source: str, doc_type: str, path: str) -> list[dict]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    step = int(_CHUNK_SIZE * 0.8)  # 20% overlap
    for i in range(0, max(1, len(words)), step):
        chunk_words = words[i : i + _CHUNK_SIZE]
        if not chunk_words:
            break
        content = " ".join(chunk_words)
        cid = hashlib.md5((source + content[:50]).encode()).hexdigest()[:12]
        chunks.append({
            "id": cid,
            "content": content,
            "metadata": {"source": source, "type": doc_type, "path": path},
        })
    return chunks


def ingest_uploaded_file(
    filename: str, content: bytes, refresh: bool = False
) -> dict:
    """Ingest an uploaded file. Returns metadata dict."""
    _ensure_dirs()
    safe_name = re.sub(r"[^\w.\-]", "_", filename)
    raw_path = _LANDING / safe_name
    raw_path.write_bytes(content)

    md_text = _to_markdown(filename, content)
    md_path = _STANDARD / (Path(safe_name).stem + ".md")
    md_path.write_text(md_text, encoding="utf-8")

    source = f"Upload: {filename}"
    new_chunks = _chunk_text(md_text, source, "upload", str(md_path.relative_to(_ROOT)))

    if refresh:
        existing = _load_chunks()
        # Remove old chunks from same source
        existing = [c for c in existing if c.get("metadata", {}).get("source") != source]
        existing.extend(new_chunks)
        _save_chunks(existing)

    return {
        "title": filename,
        "markdown_path": str(md_path),
        "chunks_added": len(new_chunks),
        "source": source,
    }


def ingest_url(url: str, refresh: bool = False) -> dict:
    """Fetch a URL, convert to markdown, and ingest. Returns metadata dict."""
    _ensure_dirs()
    import urllib.request

    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 DrugLawBot/1.0"}
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        content = resp.read()
        content_type = resp.headers.get("Content-Type", "")

    # Detect encoding
    charset = "utf-8"
    if "charset=" in content_type:
        charset = content_type.split("charset=")[-1].strip().split(";")[0].strip()

    text = content.decode(charset, errors="replace")

    # Extract title
    title_match = re.search(r"<title[^>]*>([^<]+)</title>", text, re.I)
    title = title_match.group(1).strip() if title_match else url

    # Strip HTML
    md_text = re.sub(r"<[^>]+>", " ", text)
    md_text = re.sub(r"&[a-z]+;", " ", md_text)
    md_text = re.sub(r"\s+", " ", md_text).strip()

    slug = re.sub(r"[^\w]", "_", url)[:60]
    md_path = _STANDARD / f"{slug}.md"
    md_path.write_text(md_text, encoding="utf-8")
    raw_path = _LANDING / f"{slug}.html"
    raw_path.write_bytes(content)

    source = title
    new_chunks = _chunk_text(md_text, source, "web", str(md_path.relative_to(_ROOT)))

    if refresh:
        existing = _load_chunks()
        existing = [c for c in existing if c.get("metadata", {}).get("source") != source]
        existing.extend(new_chunks)
        _save_chunks(existing)

    return {
        "title": title,
        "markdown_path": str(md_path),
        "chunks_added": len(new_chunks),
        "source": source,
    }


def refresh_index() -> dict:
    """Rebuild local_chunks.json from all standardized markdown files."""
    _ensure_dirs()
    chunks: list[dict] = []
    for md_file in _STANDARD.glob("**/*.md"):
        text = md_file.read_text(encoding="utf-8", errors="replace")
        source = md_file.stem.replace("_", " ")
        new = _chunk_text(text, source, "upload", str(md_file.relative_to(_ROOT)))
        chunks.extend(new)
    # Keep seed/legal chunks from original local_chunks.json that weren't from uploads
    seed = _load_chunks()
    non_upload = [
        c for c in seed
        if c.get("metadata", {}).get("type") in ("legal", "news", "system")
    ]
    all_chunks = non_upload + chunks
    _save_chunks(all_chunks)
    return {"documents": len(set(c.get("metadata", {}).get("source") for c in all_chunks)), "chunks": len(all_chunks)}
