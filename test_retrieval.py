"""
Tests for the Milestone 4 embedding + retrieval layer.

Two groups:
  1. CORRECTNESS — the vector store and retrieve() behave as specified:
     every chunk is indexed, top-k is honored, metadata round-trips, and
     distances are well-formed and sorted.
  2. QUALITY     — does semantic search actually surface the right chunks for
     the eval queries? Thresholds here are calibrated to observed behavior
     (see index.py's CLI output), not idealized. They encode the Milestone 4
     checkpoint: top results should be on-topic and reasonably close.

Building the index re-embeds all chunks, so it runs once per session.

Run:  python -m pytest test_retrieval.py -v
"""

import json

import pytest

import index
from pipeline import OUTPUT_PATH as CHUNKS_PATH


@pytest.fixture(scope="session", autouse=True)
def _built_index():
    """Build the Chroma collection once before any test runs."""
    index.build_index()


@pytest.fixture(scope="session")
def chunk_count():
    return len(json.loads(CHUNKS_PATH.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# 1. Correctness
# ---------------------------------------------------------------------------
def test_all_chunks_indexed(chunk_count):
    assert index.get_collection().count() == chunk_count


def test_retrieve_honors_k():
    for k in (1, 3, 5, 8):
        assert len(index.retrieve("apartment parking", k=k)) == k


def test_results_have_metadata():
    for r in index.retrieve("is the wilde a bad place to live?", k=5):
        assert r.source.endswith(".txt"), f"bad source: {r.source}"
        assert r.url.startswith("http"), f"bad url: {r.url}"
        assert r.title.strip(), "missing title"
        assert r.chunk_index >= 0
        assert r.text.strip(), "empty result text"


def test_distances_sorted_ascending():
    dists = [r.distance for r in index.retrieve("broken elevator", k=8)]
    assert dists == sorted(dists), f"distances not ascending: {dists}"


def test_distances_in_cosine_range():
    for r in index.retrieve("safety crime near campus", k=8):
        assert 0.0 <= r.distance <= 2.0, f"distance out of range: {r.distance}"


# ---------------------------------------------------------------------------
# 2. Quality — retrieval relevance on the eval queries
# ---------------------------------------------------------------------------
def test_top_hit_is_on_topic():
    """Every eval query's best match should be a reasonably close hit
    (cosine distance < 0.6 — weak matches sit well above this)."""
    weak = []
    for query, _ in index.EVAL_QUERIES:
        best = index.retrieve(query, k=5)[0].distance
        if best >= 0.60:
            weak.append((query[:45], round(best, 3)))
    assert not weak, f"queries with no close match: {weak}"


def test_mean_top_distance_is_strong():
    bests = [index.retrieve(q, k=5)[0].distance for q, _ in index.EVAL_QUERIES]
    mean = sum(bests) / len(bests)
    assert mean < 0.55, f"mean top-1 distance too high: {mean:.3f}"
