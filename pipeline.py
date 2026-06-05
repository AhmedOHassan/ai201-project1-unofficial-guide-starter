"""
Milestone 3 — Document ingestion + chunking pipeline.

Loads the 10 r/NCSU housing threads from documents/, cleans the Reddit
boilerplate (separators, vote counts, comment counts, escape artifacts), and
splits each cleaned thread into chunks using a recursive character splitter
sized in *real tokens* (via the all-MiniLM-L6-v2 tokenizer).

Spec it implements (see planning.md > Chunking Strategy):
  - Chunk size:  240 tokens  -> CHUNK_SIZE_TOKENS
  - Overlap:     40 tokens   -> CHUNK_OVERLAP_TOKENS
  - Splitter:    RecursiveCharacterTextSplitter (breaks on paragraph/line
                 boundaries first, so an apartment name stays bound to the
                 detail that follows it)
  - Post-merge:  fragments below MIN_CHUNK_TOKENS, or chunks that are just a
                 stranded "Comment by u/..." header, get glued onto a neighbor

Run it directly to build chunks, save them to data/chunks.json, and print
stats + 5 representative chunks:

    python pipeline.py
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Spec constants -------------------
CHUNK_SIZE_TOKENS = 240
CHUNK_OVERLAP_TOKENS = 40
MIN_CHUNK_TOKENS = 20

REPO_ROOT = Path(__file__).resolve().parent
DOCUMENTS_DIR = REPO_ROOT / "documents"
OUTPUT_PATH = REPO_ROOT / "data" / "chunks.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Token counting — use the embedding model's own tokenizer so "tokens" here
# means the same thing it will mean at embedding time in Milestone 4.
# ---------------------------------------------------------------------------
_TOKENIZER = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer, logging as hf_logging
        hf_logging.set_verbosity_error()
        _TOKENIZER = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    return _TOKENIZER


def count_tokens(text: str) -> int:
    """Number of word-piece tokens in `text` (no special [CLS]/[SEP] tokens)."""
    return len(_get_tokenizer().encode(text, add_special_tokens=False))


# ---------------------------------------------------------------------------
# Ingestion + cleaning
# ---------------------------------------------------------------------------
@dataclass
class Document:
    source: str          # filename, e.g. 07_the_wilde_predatory_towing.txt
    title: str           # thread title (first line of the file)
    url: str             # original Reddit URL
    text: str            # cleaned, chunk-ready text


# Lines / patterns that are pure boilerplate and carry no review content.
_SEPARATOR_RE = re.compile(r"^\s*[=\-]{5,}\s*$")
_BOILERPLATE_PREFIXES = ("Community:", "Posted by:", "Score:", "Date:", "Source:")
_COMMENTS_HEADER_RE = re.compile(r"^\s*COMMENTS\s*\(\d+\s*fetched\)\s*$", re.I)
# Comment headers look like: "[u/name] [4 pts] [Apr 21, 2023]", possibly
# indented and with a bracketed username such as "[u/[deleted]]". Use search
# (not fullmatch) and a lazy username group so the nested brackets are handled.
_COMMENT_META_RE = re.compile(
    r"\[u/(.+?)\]\s*\[\s*-?\d+\s*pts?\s*\]\s*\[([^\]]+)\]", re.I
)

# Invisible characters that survive copy/paste from Reddit and break nothing
# semantically but pollute chunks (zero-width spaces, BOM, direction marks).
_INVISIBLE_RE = re.compile(r"[​‌‍‎‏﻿]")

# Reddit markdown escapes we want to undo so chunks read like plain prose.
_ESCAPE_FIXES = {
    r"\-": "-", r"\~": "~", r"\*": "*", r"\.": ".",
    r"\(": "(", r"\)": ")", r"\[": "[", r"\]": "]",
    r"\>": ">", r"\#": "#", r"\_": "_",
}


def _extract_metadata(raw: str, filename: str) -> tuple[str, str]:
    """Pull the thread title and source URL out of the file header block."""
    lines = raw.splitlines()
    title = next((ln.strip() for ln in lines if ln.strip()), filename)
    url_match = re.search(r"^Source:\s*(\S+)", raw, re.M)
    url = url_match.group(1).strip() if url_match else ""
    return title, url


def clean_text(raw: str) -> str:
    """Strip Reddit boilerplate and normalize escapes/entities.

    Keeps: the post body and every comment's text (the substantive content).
    Removes: separator rules, the Community/Posted by/Score header lines,
             the 'COMMENTS (N fetched)' banner, and per-comment vote counts.
    Comment headers are rewritten to a clean, readable attribution line.
    """
    # Decode HTML entities first (&amp; &#39; &nbsp; ...).
    text = html.unescape(raw)
    for bad, good in _ESCAPE_FIXES.items():
        text = text.replace(bad, good)
    text = _INVISIBLE_RE.sub("", text).replace(" ", " ")

    out_lines: list[str] = []
    for line in text.splitlines():
        if _SEPARATOR_RE.match(line):
            continue
        if _COMMENTS_HEADER_RE.match(line):
            continue
        if line.strip().startswith(_BOILERPLATE_PREFIXES):
            continue

        meta = _COMMENT_META_RE.search(line)
        if meta:
            user, date = meta.group(1).strip(), meta.group(2).strip()
            out_lines.append(f"Comment by u/{user} ({date}):")
            continue

        out_lines.append(line.rstrip())

    cleaned = "\n".join(out_lines)
    # Collapse 3+ blank lines down to a single blank line.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def load_documents(documents_dir: Path = DOCUMENTS_DIR) -> list[Document]:
    """Load every .txt thread from the documents folder and clean it."""
    docs: list[Document] = []
    for path in sorted(documents_dir.glob("*.txt")):
        raw = path.read_text(encoding="utf-8")
        title, url = _extract_metadata(raw, path.name)
        cleaned = clean_text(raw)
        # Keep the thread title at the top of the body so the apartment being
        # discussed is present in the document text, not just the metadata.
        body = f"{title}\n\n{cleaned}" if not cleaned.startswith(title) else cleaned
        docs.append(Document(source=path.name, title=title, url=url, text=body))
    return docs


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def _make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        length_function=count_tokens,
        # Prefer to break on blank lines, then single lines, then sentences,
        # then words — so we cut at natural boundaries, not mid-word.
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )


# A chunk that is nothing but comment-attribution headers (the splitter broke
# right after a "Comment by ..." line, stranding it from its comment text).
_HEADER_ONLY_RE = re.compile(r"^(?:Comment by u/.+?\([^)]*\):\s*)+$")


def _is_fragment(piece: str) -> bool:
    """A piece too small or header-only to stand on its own."""
    return count_tokens(piece) < MIN_CHUNK_TOKENS or bool(
        _HEADER_ONLY_RE.match(piece.strip())
    )


def _merge_fragments(pieces: list[str]) -> list[str]:
    """Glue fragment pieces onto a neighbor so no chunk is a stranded header
    or a meaningless sliver. Merge forward (a header rejoins its comment);
    a trailing fragment merges back into the previous chunk."""
    out: list[str] = []
    i = 0
    while i < len(pieces):
        cur = pieces[i]
        while _is_fragment(cur) and i + 1 < len(pieces):
            i += 1
            cur = f"{cur}\n\n{pieces[i]}"
        if _is_fragment(cur) and out:
            out[-1] = f"{out[-1]}\n\n{cur}"
        else:
            out.append(cur)
        i += 1
    return out


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    title: str
    url: str
    chunk_index: int
    num_tokens: int


def chunk_documents(docs: list[Document]) -> list[Chunk]:
    splitter = _make_splitter()
    chunks: list[Chunk] = []
    for doc in docs:
        pieces = [p.strip() for p in splitter.split_text(doc.text) if p.strip()]
        pieces = _merge_fragments(pieces)
        for i, piece in enumerate(pieces):
            stem = doc.source.replace(".txt", "")
            chunks.append(
                Chunk(
                    id=f"{stem}__{i}",
                    text=piece,
                    source=doc.source,
                    title=doc.title,
                    url=doc.url,
                    chunk_index=i,
                    num_tokens=count_tokens(piece),
                )
            )
    return chunks


def build_chunks() -> list[Chunk]:
    return chunk_documents(load_documents())


def save_chunks(chunks: list[Chunk], path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(c) for c in chunks], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CLI: build, save, and show stats + samples for the Milestone 3 checkpoint.
# ---------------------------------------------------------------------------
def _print_report(chunks: list[Chunk]) -> None:
    import statistics

    token_counts = [c.num_tokens for c in chunks]
    per_doc: dict[str, int] = {}
    for c in chunks:
        per_doc[c.source] = per_doc.get(c.source, 0) + 1

    print("\n" + "=" * 70)
    print(f"TOTAL CHUNKS: {len(chunks)}")
    print("=" * 70)
    print(f"tokens/chunk -> min {min(token_counts)} | "
          f"median {int(statistics.median(token_counts))} | "
          f"mean {statistics.mean(token_counts):.0f} | max {max(token_counts)}")
    print("\nchunks per document:")
    for src in sorted(per_doc):
        print(f"  {per_doc[src]:>3}  {src}")

    print("\n" + "=" * 70)
    print("5 REPRESENTATIVE CHUNKS (spread across the corpus)")
    print("=" * 70)
    step = max(1, len(chunks) // 5)
    for c in chunks[::step][:5]:
        print(f"\n--- {c.id}  [{c.num_tokens} tokens]  source={c.source} ---")
        print(c.text)


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # print curly quotes etc. on Windows
    except Exception:
        pass
    chunks = build_chunks()
    save_chunks(chunks)
    _print_report(chunks)
    print(f"\nSaved {len(chunks)} chunks -> {OUTPUT_PATH.relative_to(REPO_ROOT)}")
