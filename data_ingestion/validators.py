"""Safe validation of uploaded files before any parsing happens."""
from __future__ import annotations

import re
from pathlib import Path

import pymupdf

from config import settings


class FileValidationError(ValueError):
    """Raised with a user-safe message (no paths, no stack traces)."""


SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(name: str) -> str:
    """Drop any directory part and unusual characters so a name can never escape the papers folder."""
    base = Path(name or "upload.pdf").name
    base = SAFE_NAME_RE.sub("_", base).strip("._") or "upload.pdf"
    if not base.lower().endswith(".pdf"):
        base += ".pdf"
    return base[:120]


def validate_pdf_bytes(data: bytes, file_name: str) -> int:
    """Check type, size and integrity. Returns the page count when the file is acceptable."""
    if not file_name.lower().endswith(".pdf"):
        raise FileValidationError("Only PDF files are supported.")
    size_mb = len(data) / (1024 * 1024)
    if size_mb == 0:
        raise FileValidationError("The file is empty.")
    if size_mb > settings.max_upload_mb:
        raise FileValidationError(f"The file is larger than the {settings.max_upload_mb} MB limit.")
    if not data[:1024].lstrip().startswith(b"%PDF-"):
        raise FileValidationError("The file does not look like a valid PDF.")
    try:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            if doc.needs_pass:
                raise FileValidationError("Password-protected PDFs are not supported.")
            pages = doc.page_count
    except FileValidationError:
        raise
    except Exception:  # malformed PDF - never leak the parser error to the user
        raise FileValidationError("The PDF could not be opened. It may be damaged.")
    if pages == 0:
        raise FileValidationError("The PDF has no pages.")
    if pages > settings.max_pages:
        raise FileValidationError(f"The PDF has more than {settings.max_pages} pages.")
    return pages
