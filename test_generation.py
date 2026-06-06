"""
Tests for the Milestone 5 grounded-generation layer.

Two groups:
  1. UNIT     — no network. Verify the grounding scaffolding: context is
     source-labeled, citations are de-duplicated from real chunk metadata,
     the system prompt actually enforces grounding, and refusal detection
     works. These guarantee that *if* the model misbehaves we still don't
     fabricate citations.
  2. INTEGRATION — calls Groq's live API; skipped automatically when
     GROQ_API_KEY isn't set. Checks an in-scope query is answered with
     sources and an out-of-scope query is refused with no sources.

Run:  python -m pytest test_generation.py -v
"""

import os

import pytest

import query
from index import Result

# query.load_dotenv() already ran on import, so .env is loaded here.
needs_api = pytest.mark.skipif(not query.os.getenv("GROQ_API_KEY"),
                               reason="GROQ_API_KEY not set")


def _fake_results():
    return [
        Result(text="The Wilde towed my legally parked car and damaged it.",
               source="07_the_wilde_predatory_towing.txt", title="The Wilde",
               url="http://x/7", chunk_index=2, distance=0.41),
        Result(text="College Inn parking is an extra $40/mo.",
               source="05_spotting_fake_corporate_reviews.txt", title="Mega",
               url="http://x/5", chunk_index=9, distance=0.46),
        Result(text="More Wilde complaints about management.",
               source="07_the_wilde_predatory_towing.txt", title="The Wilde",
               url="http://x/7", chunk_index=3, distance=0.50),
    ]


# ---------------------------------------------------------------------------
# 1. Unit
# ---------------------------------------------------------------------------
def test_context_is_source_labeled():
    ctx = query._build_context(_fake_results())
    assert "(source: 07_the_wilde_predatory_towing.txt)" in ctx
    assert "towed my legally parked car" in ctx
    assert "[1]" in ctx and "[2]" in ctx and "[3]" in ctx


def test_unique_sources_dedup_preserves_order():
    srcs = query._unique_sources(_fake_results())
    assert [s["source"] for s in srcs] == [
        "07_the_wilde_predatory_towing.txt",
        "05_spotting_fake_corporate_reviews.txt",
    ]
    assert all("url" in s for s in srcs)


def test_system_prompt_enforces_grounding():
    p = query.SYSTEM_PROMPT
    assert query.REFUSAL in p, "prompt must define the exact refusal string"
    assert "ONLY" in p, "prompt must restrict the model to the context"


def test_refusal_detection():
    assert query.is_refusal("I don't have enough information on that.")
    assert query.is_refusal("i don't have enough information on that")  # no period
    assert not query.is_refusal("The Wilde tows cars, per student reviews.")


# ---------------------------------------------------------------------------
# 2. Integration (live Groq) — skipped without an API key
# ---------------------------------------------------------------------------
@needs_api
def test_in_scope_query_is_grounded_and_cited():
    r = query.ask("What's the window or fire-safety problem at The Standard?")
    assert not r["refused"], f"unexpected refusal: {r['answer']}"
    assert r["sources"], "in-scope answer must carry source citations"
    cited = {s["source"] for s in r["sources"]}
    assert "09_the_standard_window_safety_hazards.txt" in cited, cited


@needs_api
def test_out_of_scope_query_refuses_without_sources():
    r = query.ask("What is the capital of France?")
    assert r["refused"], f"should have refused, got: {r['answer']}"
    assert r["sources"] == [], "a refusal must not cite any sources"
