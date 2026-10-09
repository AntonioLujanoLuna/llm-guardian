"""Segmentation: stability, chunking, overlap and the truncation contract."""

from __future__ import annotations

import hashlib
import itertools

from llm_guardian.models import SegmentInput
from llm_guardian.provenance import Provenance, TrustLevel
from llm_guardian.segments import Segmenter, digest_of, make_segment_id
from llm_guardian.settings import ChunkSettings


def make_segmenter(max_chars: int = 64, overlap: int = 0, max_chunks: int = 8) -> Segmenter:
    return Segmenter(
        ChunkSettings(
            max_segment_chars=max_chars, overlap_chars=overlap, max_chunks_per_segment=max_chunks
        )
    )


def test_digest_is_sha256() -> None:
    assert digest_of("abc") == hashlib.sha256(b"abc").hexdigest()


def test_short_text_is_one_segment() -> None:
    segments = make_segmenter().segment("c1", [SegmentInput(text="hi", provenance=Provenance.USER)])
    assert len(segments) == 1
    assert segments[0].index == 0
    assert segments[0].digest == digest_of("hi")
    assert segments[0].trust is TrustLevel.OPERATOR
    assert segments[0].char_start == 0
    assert segments[0].char_end == 2
    assert segments[0].truncated is False


def test_segment_id_stable_across_retries() -> None:
    inputs = [SegmentInput(text="stable text", provenance=Provenance.USER)]
    first = make_segmenter().segment("conv-1", inputs)
    second = make_segmenter().segment("conv-1", inputs)
    assert [s.segment_id for s in first] == [s.segment_id for s in second]
    assert first[0].segment_id == make_segment_id("conv-1", 0, digest_of("stable text"))


def test_long_text_splits_with_overlap() -> None:
    text = " ".join(f"word{i}" for i in range(300))
    segments = make_segmenter(max_chars=64, overlap=10, max_chunks=100).segment(
        "c1", [SegmentInput(text=text, provenance=Provenance.USER)]
    )
    assert len(segments) > 1
    assert all(len(s.text) <= 64 for s in segments)
    # each chunk overlaps the previous by about overlap_chars
    for previous, current in itertools.pairwise(segments):
        assert current.char_start is not None and previous.char_end is not None
        assert current.char_start < previous.char_end
    assert segments[-1].char_end == len(text)


def test_prefers_newline_boundary() -> None:
    text = "a" * 40 + "\n" + "b" * 40
    segments = make_segmenter(max_chars=64, overlap=0).segment(
        "c1", [SegmentInput(text=text, provenance=Provenance.USER)]
    )
    assert segments[0].text.endswith("\n")
    assert len(segments[0].text) == 41


def test_falls_back_to_space_boundary() -> None:
    text = "x" * 40 + " " + "y" * 40
    segments = make_segmenter(max_chars=64, overlap=0).segment(
        "c1", [SegmentInput(text=text, provenance=Provenance.USER)]
    )
    assert segments[0].text == "x" * 40 + " "


def test_no_boundary_uses_full_window() -> None:
    text = "z" * 200
    segments = make_segmenter(max_chars=64, overlap=0).segment(
        "c1", [SegmentInput(text=text, provenance=Provenance.USER)]
    )
    assert all(len(s.text) == 64 for s in segments[:-1])


def test_max_chunks_clips_and_marks_truncated() -> None:
    text = " ".join(f"w{i}" for i in range(500))
    segments = make_segmenter(max_chars=64, overlap=0, max_chunks=2).segment(
        "c1", [SegmentInput(text=text, provenance=Provenance.USER)]
    )
    assert len(segments) == 2
    assert segments[-1].truncated is True
    assert segments[-1].char_end is not None and segments[-1].char_end < len(text)


def test_truncated_input_flag_propagates() -> None:
    segments = make_segmenter().segment(
        "c1", [SegmentInput(text="short", provenance=Provenance.USER, truncated=True)]
    )
    assert segments[0].truncated is True


def test_char_start_offset_applied() -> None:
    segments = make_segmenter().segment(
        "c1",
        [SegmentInput(text="hello", provenance=Provenance.USER, char_start=1000)],
    )
    assert segments[0].char_start == 1000
    assert segments[0].char_end == 1005


def test_multiple_inputs_get_global_indexes() -> None:
    inputs = [
        SegmentInput(text="one", provenance=Provenance.USER),
        SegmentInput(text="two", provenance=Provenance.RETRIEVED_CHUNK),
    ]
    segments = make_segmenter().segment("c1", inputs)
    assert [s.index for s in segments] == [0, 1]
    assert segments[1].provenance is Provenance.RETRIEVED_CHUNK


def test_duplicate_text_produces_same_digest() -> None:
    inputs = [
        SegmentInput(text="same", provenance=Provenance.USER),
        SegmentInput(text="same", provenance=Provenance.USER),
    ]
    segments = make_segmenter().segment("c1", inputs)
    assert segments[0].digest == segments[1].digest
    assert segments[0].segment_id != segments[1].segment_id  # index differs


def test_empty_input_list() -> None:
    assert make_segmenter().segment("c1", []) == []
