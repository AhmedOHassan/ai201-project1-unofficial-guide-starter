"""
Tests for the Milestone 3 chunking pipeline.

Two groups:
  1. CORRECTNESS  — the chunks must be clean and well-formed. These should
     always pass; a failure means ingestion/cleaning is broken.
  2. STRATEGY     — is the chosen chunk size actually good for *this* corpus
     and *this* embedding model? These encode the rules of thumb from the
     instructions (>=50 chunks for 10 docs) and the hard limit of the
     embedding model (all-MiniLM-L6-v2 only encodes 256 tokens). If they
     fail, the chunking strategy needs modification, not the cleaning.

Run:  python -m pytest test_chunks.py -v
"""

import re

import pytest

import pipeline

EMBED_MAX_TOKENS = 256

KNOWN_TOPICS = [
    "wilde", "valentine", "standard", "university woods", "college inn",
    "stanhope", "campus crossing", "village green", "gorman", "trinity",
    "avent ferry", "hillsborough", "wolfline", "parking", "rent", "maintenance",
    "mold", "roach", "towed", "tow", "sublease", "bus", "fee", "security",
]


@pytest.fixture(scope="module")
def chunks():
    return pipeline.build_chunks()


# ---------------------------------------------------------------------------
# 1. Correctness — cleaning and structure
# ---------------------------------------------------------------------------
def test_chunks_exist(chunks):
    assert len(chunks) > 0, "pipeline produced no chunks"


def test_no_empty_chunks(chunks):
    assert all(c.text.strip() for c in chunks), "found empty/whitespace chunk"


def test_unique_ids(chunks):
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk ids are not unique"


def test_metadata_complete(chunks):
    for c in chunks:
        assert c.source.endswith(".txt"), f"{c.id}: bad source"
        assert c.title.strip(), f"{c.id}: missing title"
        assert c.url.startswith("http"), f"{c.id}: missing/invalid url"
        assert c.chunk_index >= 0, f"{c.id}: bad chunk_index"
        assert c.num_tokens > 0, f"{c.id}: non-positive token count"


def test_no_cleaning_artifacts(chunks):
    """No leftover Reddit boilerplate, HTML entities, or invisible chars."""
    artifacts = {
        "vote count":     re.compile(r"\[\s*-?\d+\s*pts?\s*\]", re.I),
        "separator rule": re.compile(r"[=\-]{5,}"),
        "html entity":    re.compile(r"&(amp|nbsp|gt|lt|quot|#\d+);"),
        "comments banner":re.compile(r"COMMENTS\s*\(\d+\s*fetched\)", re.I),
        "score line":     re.compile(r"^\s*Score:\s", re.M),
        "source line":    re.compile(r"^\s*Source:\s*http", re.M),
        "zero-width char":re.compile(r"[​‌‍‎‏﻿]"),
    }
    for c in chunks:
        for name, rx in artifacts.items():
            assert not rx.search(c.text), f"{c.id}: leftover {name}: {c.text[:80]!r}"


def test_no_fragment_chunks(chunks):
    """No chunk should be a tiny sliver or a stranded 'Comment by ...' header
    with no body -- those carry no standalone meaning."""
    frags = [(c.id, c.num_tokens) for c in chunks
             if c.num_tokens < pipeline.MIN_CHUNK_TOKENS
             or pipeline._HEADER_ONLY_RE.match(c.text.strip())]
    assert not frags, f"fragment/header-only chunks survived: {frags}"


def test_splitter_respects_size(chunks):
    """No chunk should blow past the configured chunk size (small tolerance)."""
    ceiling = int(pipeline.CHUNK_SIZE_TOKENS * 1.10)
    too_big = [(c.id, c.num_tokens) for c in chunks if c.num_tokens > ceiling]
    assert not too_big, f"chunks exceed configured size {ceiling}: {too_big}"


# ---------------------------------------------------------------------------
# 2. Strategy — is the chunk size right for this corpus + embedding model?
# ---------------------------------------------------------------------------
def test_chunk_count_in_recommended_range(chunks):
    """Instructions: <50 chunks for 10 docs usually means chunks too large."""
    assert 50 <= len(chunks) <= 2000, (
        f"got {len(chunks)} chunks; outside the 50-2000 sweet spot for 10 docs "
        f"(too few -> chunks too large to match specific queries)"
    )


def test_chunks_fit_embedding_window(chunks):
    """all-MiniLM-L6-v2 only embeds the first 256 tokens; longer chunks lose
    their tail at embedding time. Most chunks should fit the window."""
    over = [(c.id, c.num_tokens) for c in chunks if c.num_tokens > EMBED_MAX_TOKENS]
    frac = len(over) / len(chunks)
    assert frac <= 0.10, (
        f"{len(over)}/{len(chunks)} ({frac:.0%}) chunks exceed the {EMBED_MAX_TOKENS}-token "
        f"embedding window and will be truncated: {over[:5]}"
    )


def test_chunks_carry_topic_context(chunks):
    """A retrievable chunk should mention something concrete about housing,
    not be pure chit-chat. Allow a few, but most must carry a topic word."""
    def has_topic(c):
        blob = (c.text + " " + c.title).lower()
        return any(t in blob for t in KNOWN_TOPICS)

    contextful = [c for c in chunks if has_topic(c)]
    frac = len(contextful) / len(chunks)
    assert frac >= 0.85, (
        f"only {frac:.0%} of chunks contain a recognizable housing topic; "
        f"chunks may be fragmented or off-topic"
    )
