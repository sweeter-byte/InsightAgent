"""URL ingestion loader using httpx + BeautifulSoup.

Performs an HTTP GET, follows redirects, strips script/style/noscript tags,
and extracts the visible text and title from the HTML response.

No SSRF protection, no dynamic rendering, no auth, no readability algorithm —
all explicitly out of Chapter 3 scope.
"""

from __future__ import annotations

import httpx
from bs4 import BeautifulSoup

from insight_agent.ingestion.errors import IngestionError
from insight_agent.ingestion.models import Document, SourceType

_USER_AGENT = "InsightAgent/0.1"


def load_url(url: str, *, timeout: float = 30.0) -> list[Document]:
    """Fetch a URL and return its visible text content as a single Document.

    Args:
        url:     the HTTP/HTTPS URL to fetch.
        timeout: per-request timeout in seconds.

    Returns:
        ``[Document]`` with ``source_type=URL``.

    Raises:
        IngestionError: on network errors, non-2xx status, or parse failures.
    """

    try:
        response = httpx.get(
            url,
            follow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": _USER_AGENT},
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise IngestionError(
            f"HTTP {exc.response.status_code} for {url}"
        ) from exc
    except httpx.RequestError as exc:
        raise IngestionError(f"Request failed for {url}: {exc}") from exc

    try:
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove non-visible elements
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        title: str = ""
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            title = title_tag.string.strip()

        body_text: str = soup.get_text(separator="\n", strip=True)
    except Exception as exc:
        raise IngestionError(f"Failed to parse HTML from {url}: {exc}") from exc

    content_type: str = response.headers.get("content-type", "")

    return [
        Document(
            content=body_text,
            source=str(response.url),
            source_type=SourceType.URL,
            metadata={
                "title": title,
                "final_url": str(response.url),
                "content_type": content_type,
            },
        )
    ]
