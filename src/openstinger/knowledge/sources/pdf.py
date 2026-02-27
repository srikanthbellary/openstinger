"""PDF source extractor — extract text from a local PDF file."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


async def extract(file_path: str | Path) -> str:
    """
    Extract text from a PDF file.

    Tries pdfplumber first (better layout), falls back to pypdf.
    Requires: pdfplumber or pypdf (optional dependencies).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    # Try pdfplumber
    try:
        import pdfplumber  # type: ignore[import]

        pages: list[str] = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text.strip())
        logger.debug("Extracted %d pages from %s (pdfplumber)", len(pages), path.name)
        return "\n\n".join(pages)
    except ImportError:
        pass

    # Fallback: pypdf
    try:
        from pypdf import PdfReader  # type: ignore[import]

        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
        logger.debug("Extracted %d pages from %s (pypdf)", len(pages), path.name)
        return "\n\n".join(pages)
    except ImportError:
        pass

    raise ImportError(
        "PDF extraction requires pdfplumber or pypdf: "
        "pip install pdfplumber  OR  pip install pypdf"
    )
