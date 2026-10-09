"""Incremental segmentation.

The gateway decides *which* parts of a request are new or modified; this module turns those parts
into the :class:`~llm_guardian.models.Segment` objects the pipeline walks, splitting anything long
into overlapping chunks that break on a newline or space when one is available.

Two properties matter more than the chunking itself:

* A ``segment_id`` is stable across retries -- it is a function of the conversation, the position
  and the exact content digest -- so a repeated request maps to the same ids.
* Bounded inspection is never silent. When ``max_chunks_per_segment`` clips content, the last kept
  chunk is marked ``truncated`` and the service surfaces it as a warning; a truncated scan is not
  evidence of safety.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from .models import Segment, SegmentInput
from .provenance import trust_of
from .settings import ChunkSettings


def digest_of(text: str) -> str:
    """SHA-256 of the exact bytes that will be scanned."""
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def make_segment_id(conversation_id: str | None, index: int, digest: str) -> str:
    """A stable id: conversation, position and content digest, in that order."""
    conversation = conversation_id or "anon"
    return f"{conversation}:{index}:{digest[:16]}"


class Segmenter:
    """Builds segments from gateway-submitted inputs, honouring the chunking settings."""

    def __init__(self, settings: ChunkSettings) -> None:
        self._settings = settings

    def segment(self, conversation_id: str | None, inputs: Sequence[SegmentInput]) -> list[Segment]:
        segments: list[Segment] = []
        index = 0
        for item in inputs:
            for chunk_text, start, end, truncated in self._chunks(item):
                digest = digest_of(chunk_text)
                segments.append(
                    Segment(
                        segment_id=make_segment_id(conversation_id, index, digest),
                        index=index,
                        digest=digest,
                        text=chunk_text,
                        provenance=item.provenance,
                        trust=trust_of(item.provenance),
                        char_start=start,
                        char_end=end,
                        truncated=truncated,
                    )
                )
                index += 1
        return segments

    def _chunks(self, item: SegmentInput) -> list[tuple[str, int, int, bool]]:
        text = item.text
        base = item.char_start or 0
        raw = _split(
            text,
            max_chars=self._settings.max_segment_chars,
            overlap=self._settings.overlap_chars,
        )
        clipped = len(raw) > self._settings.max_chunks_per_segment
        raw = raw[: self._settings.max_chunks_per_segment]

        chunks: list[tuple[str, int, int, bool]] = []
        last = len(raw) - 1
        for position, (chunk_text, start, end) in enumerate(raw):
            at_end = end >= len(text)
            truncated = item.truncated or (position == last and (clipped or not at_end))
            if position == last and not at_end:
                truncated = True
            chunks.append((chunk_text, base + start, base + end, truncated))
        return chunks


def _split(text: str, *, max_chars: int, overlap: int) -> list[tuple[str, int, int]]:
    """Split ``text`` into windows of at most ``max_chars`` with ``overlap`` carried across."""
    if len(text) <= max_chars:
        return [(text, 0, len(text))]

    chunks: list[tuple[str, int, int]] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        if end < length:
            end = _preferred_break(text, start, end, max_chars)
        chunks.append((text[start:end], start, end))
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return chunks


def _preferred_break(text: str, start: int, end: int, max_chars: int) -> int:
    """Pull ``end`` back to the last newline (preferred) or space in the window's tail."""
    window = text[start:end]
    threshold = max_chars // 2
    newline = window.rfind("\n")
    if newline >= threshold:
        return start + newline + 1
    space = max(window.rfind(" "), window.rfind("\t"))
    if space >= threshold:
        return start + space + 1
    return end
