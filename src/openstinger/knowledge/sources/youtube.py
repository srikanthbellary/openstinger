"""YouTube source extractor — fetch transcript from a YouTube video."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_YT_URL_PATTERNS = [
    r"(?:v=|youtu\.be/)([A-Za-z0-9_\-]{11})",
    r"youtube\.com/embed/([A-Za-z0-9_\-]{11})",
]


def _extract_video_id(source: str) -> str:
    """Extract the 11-char video ID from a YouTube URL or bare ID."""
    source = source.strip()
    # Already a bare ID
    if re.fullmatch(r"[A-Za-z0-9_\-]{11}", source):
        return source
    for pattern in _YT_URL_PATTERNS:
        m = re.search(pattern, source)
        if m:
            return m.group(1)
    raise ValueError(f"Cannot extract YouTube video ID from: {source!r}")


async def extract(source: str, languages: list[str] | None = None) -> str:
    """
    Fetch the transcript for a YouTube video.

    *source* can be a full YouTube URL or a bare 11-char video ID.
    *languages* is a preference list, e.g. ["en", "en-US"] (default: ["en"]).

    Requires: youtube-transcript-api (optional dependency).
    Install: pip install youtube-transcript-api
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore[import]
        from youtube_transcript_api._errors import (  # type: ignore[import]
            NoTranscriptFound,
            TranscriptsDisabled,
        )
    except ImportError as exc:
        raise ImportError(
            "youtube-transcript-api is required: pip install youtube-transcript-api"
        ) from exc

    video_id = _extract_video_id(source)
    langs = languages or ["en", "en-US", "en-GB"]

    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        # Try requested languages first
        try:
            transcript = transcript_list.find_transcript(langs)
        except NoTranscriptFound:
            # Fall back to any available transcript
            transcript = next(iter(transcript_list))

        entries = transcript.fetch()
        text = " ".join(entry["text"] for entry in entries)
        logger.debug("Fetched YouTube transcript for %s (%d entries)", video_id, len(entries))
        return text.strip()

    except TranscriptsDisabled:
        raise ValueError(f"Transcripts are disabled for video: {video_id}")
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch YouTube transcript for {video_id}: {exc}") from exc
