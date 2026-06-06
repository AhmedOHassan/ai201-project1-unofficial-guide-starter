"""
Milestone 4 — Embedding + vector store + retrieval.

Embeds the Milestone 3 chunks with all-MiniLM-L6-v2, stores them in a local
ChromaDB collection (with source/title/url/chunk_index metadata for later
attribution), and exposes retrieve() for semantic search.

The collection uses cosine distance, so scores run 0 (identical) -> 2
(opposite); for this corpus, anything under ~0.5 on the top hit is a strong
match.

Build the index, then run the eval queries to inspect retrieval quality:

    python index.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from pipeline import OUTPUT_PATH as CHUNKS_PATH, EMBEDDING_MODEL, REPO_ROOT

CHROMA_DIR = REPO_ROOT / "data" / "chroma"
COLLECTION_NAME = "ncsu_housing"
DEFAULT_TOP_K = 5

# The 5 evaluation queries from planning.md, each paired with the source file(s).
EVAL_QUERIES: list[tuple[str, tuple[str, ...]]] = [
    ("What are the hidden fees and parking costs at the budget complexes students mention?",
     ("03_hidden_fees_parking_costs.txt", "05_spotting_fake_corporate_reviews.txt")),
    ("Is there a safety concern reported near Avent Ferry and Socket Dr?",
     ("02_avent_ferry_safety_prowler.txt",)),
    ("Which management company do students say tows cars or damages vehicles?",
     ("07_the_wilde_predatory_towing.txt",)),
    ("What's the window or fire-safety problem students raise about The Standard?",
     ("09_the_standard_window_safety_hazards.txt",)),
    ("At Valentine Commons, what infrastructure problems do reviewers report?",
     ("10_valentine_commons_infrastructure.txt",)),
]


# ---------------------------------------------------------------------------
# Embedding model (loaded once, lazily)
# ---------------------------------------------------------------------------
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """Encode texts into normalized vectors (normalized -> cosine distance)."""
    return get_model().encode(
        texts, normalize_embeddings=True, show_progress_bar=False
    ).tolist()


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def _load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"{CHUNKS_PATH} not found — run `python pipeline.py` first."
        )
    return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))


def build_index() -> int:
    """(Re)build the Chroma collection from data/chunks.json. Returns count."""
    chunks = _load_chunks()

    # Start clean so re-running doesn't pile up duplicate ids.
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embed([c["text"] for c in chunks]),
        metadatas=[
            {
                "source": c["source"],
                "title": c["title"],
                "url": c["url"],
                "chunk_index": c["chunk_index"],
            }
            for c in chunks
        ],
    )
    return collection.count()


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
@dataclass
class Result:
    text: str
    source: str
    title: str
    url: str
    chunk_index: int
    distance: float


def retrieve(query: str, k: int = DEFAULT_TOP_K) -> list[Result]:
    """Return the top-k most semantically similar chunks to `query`."""
    collection = get_collection()
    res = collection.query(
        query_embeddings=embed([query]),
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    out: list[Result] = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        out.append(
            Result(
                text=doc,
                source=meta["source"],
                title=meta["title"],
                url=meta["url"],
                chunk_index=meta["chunk_index"],
                distance=dist,
            )
        )
    return out


# ---------------------------------------------------------------------------
# CLI: build the index and inspect retrieval on the eval queries.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    count = build_index()
    print(f"Indexed {count} chunks into '{COLLECTION_NAME}' at "
          f"{CHROMA_DIR.relative_to(REPO_ROOT)}\n")

    for query, expected in EVAL_QUERIES:
        print("=" * 78)
        print(f"QUERY: {query}")
        print(f"(expected source(s): {', '.join(expected)})")
        print("-" * 78)
        for i, r in enumerate(retrieve(query), 1):
            hit = "  <-- expected" if r.source in expected else ""
            preview = r.text.replace("\n", " ")[:140]
            print(f"{i}. dist={r.distance:.3f}  {r.source}{hit}")
            print(f"   {preview}...")
        print()
