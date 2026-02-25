"""URL source extractor — fetch HTML and extract readable text."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; OpenStinger/0.3; +https://github.com/openstinger)"
    )
}


async def extract(url: str, timeout: float = 30.0) -> str:
    """
    Fetch a URL and return the page's readable text.

    Requires: httpx and beautifulsoup4 (optional dependencies).
    Falls back to raw text extraction if beautifulsoup4 is unavailable.
    """
    try:
        import httpx
    except ImportError as exc:
        raise ImportError(
            "httpx is required for URL extraction: pip install httpx"
        ) from exc

    async with httpx.AsyncClient(headers=_HEADERS, follow_redirects=True, timeout=timeout) as client:
        response = await client.get(url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        raw_text = response.text

    if "html" in content_type.lower():
        return _extract_html(raw_text, url)
    return raw_text.strip()


def _extract_html(html: str, url: str) -> str:
    """Extract readable text from HTML. Uses BeautifulSoup if available."""
    try:
        from bs4 import BeautifulSoup  # type: ignore[import]

        soup = BeautifulSoup(html, "html.parser")
        # Remove non-content tags
        for tag in soup(["script", "style", "nav", "footer", "header",
                          "aside", "form", "noscript", "iframe"]):
            tag.decompose()
        # Try to find main content block
        main = (
            soup.find("main")
            or soup.find("article")
            or soup.find(id="content")
            or soup.find(id="main")
            or soup.body
        )
        text = (main or soup).get_text(separator="\n", strip=True)
        # Collapse excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
    except ImportError:
        logger.debug("beautifulsoup4 not installed — using regex HTML strip for %s", url)
        return _strip_html_regex(html)


def _strip_html_regex(html: str) -> str:
    """Minimal HTML tag stripper using regex (fallback)."""
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&\w+;", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
