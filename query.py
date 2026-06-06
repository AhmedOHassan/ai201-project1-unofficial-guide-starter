"""
Milestone 5 — Grounded generation.

Ties retrieval (index.retrieve) to Groq's llama-3.3-70b-versatile. The model is
instructed to answer ONLY from the retrieved chunks; if they don't contain the
answer it must return a fixed refusal sentinel. Source attribution is NOT left
to the model — it's built programmatically from the metadata of the chunks that
were actually retrieved, so every non-refused answer carries real citations.

    from query import ask
    result = ask("Which complex tows cars?")
    print(result["answer"], result["sources"])
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from index import retrieve, Result, DEFAULT_TOP_K

load_dotenv()

LLM_MODEL = "llama-3.3-70b-versatile"

# Exact string the model must emit when the context can't answer the question.
# We detect it verbatim to suppress citations on a refusal.
REFUSAL = "I don't have enough information on that."

# If even the best chunk is this far away, the corpus almost certainly doesn't
# cover the question — skip the LLM call and refuse directly. Calibrated above
# the in-scope top-1 distances observed in Milestone 4 (~0.39-0.56).
RELEVANCE_GATE = 0.85

SYSTEM_PROMPT = (
    "You are The Unofficial Guide, answering questions about off-campus student "
    "housing near NC State using ONLY the student reviews provided in the "
    "context below.\n\n"
    "Rules:\n"
    "1. Use ONLY information in the provided context. Do not use any outside or "
    "general knowledge.\n"
    "2. If the context does not contain enough information to answer the "
    f"question, reply with EXACTLY this and nothing else: \"{REFUSAL}\"\n"
    "3. Do not invent apartment names, prices, or facts that are not in the "
    "context.\n"
    "4. Be concrete: quote the specific complexes, dollar amounts, and problems "
    "students actually mention.\n"
    "5. Keep the answer to a few sentences."
)


@dataclass
class Answer:
    answer: str
    sources: list[dict]      # [{"source": ..., "url": ...}] — empty on refusal
    results: list[Result]    # the retrieved chunks (for the UI / debugging)
    refused: bool


def _build_context(results: list[Result]) -> str:
    """Format retrieved chunks into a numbered, source-labeled context block."""
    blocks = []
    for i, r in enumerate(results, 1):
        blocks.append(f"[{i}] (source: {r.source})\n{r.text}")
    return "\n\n".join(blocks)


def _unique_sources(results: list[Result]) -> list[dict]:
    """Distinct sources from the retrieved chunks, preserving rank order."""
    seen, out = set(), []
    for r in results:
        if r.source not in seen:
            seen.add(r.source)
            out.append({"source": r.source, "url": r.url})
    return out


def is_refusal(text: str) -> bool:
    """True if the model emitted the refusal sentinel (punctuation-insensitive)."""
    return REFUSAL.rstrip(".").lower() in text.rstrip(".").lower()


def _get_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_key_here":
        raise RuntimeError(
            "GROQ_API_KEY is not set — copy .env.example to .env and add your key."
        )
    from groq import Groq
    return Groq(api_key=api_key)


def ask(question: str, k: int = DEFAULT_TOP_K) -> dict:
    """Retrieve, generate a grounded answer, and attach real source citations."""
    results = retrieve(question, k=k)

    if not results or results[0].distance > RELEVANCE_GATE:
        return Answer(REFUSAL, [], results, refused=True).__dict__

    context = _build_context(results)
    user_msg = f"Context:\n{context}\n\nQuestion: {question}"

    completion = _get_client().chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,  # deterministic, grounded answers
    )
    answer_text = completion.choices[0].message.content.strip()

    refused = is_refusal(answer_text)
    sources = [] if refused else _unique_sources(results)
    return Answer(answer_text, sources, results, refused).__dict__


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    demos = [
        "Which apartment complex do students say tows and damages cars?",
        "What are the parking costs at the budget complexes?",
        "What is the best pizza topping?",  # out of scope -> should refuse
    ]
    for q in demos:
        r = ask(q)
        print("=" * 78)
        print(f"Q: {q}")
        print(f"A: {r['answer']}")
        print(f"refused: {r['refused']}")
        print("sources:", ", ".join(s["source"] for s in r["sources"]) or "(none)")
        print()
