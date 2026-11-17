"""Central configuration. Every field can be overridden from the CLI."""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Similarity below which we refuse to answer ("grounding gate").
# Score scales differ per embedding family, so defaults are per backend.
# CALIBRATE these with `python -m rag.cli experiments` (it prints the scores of
# in-scope vs out-of-scope questions).
DEFAULT_MIN_SCORE = {"st": 0.50, "openai": 0.25, "gemini": 0.50, "tfidf": 0.10}

DEFAULT_EMBEDDING_MODEL = {
    "st": "BAAI/bge-small-en-v1.5",
    "openai": "text-embedding-3-small",
    "gemini": "gemini-embedding-001",
    "tfidf": "tfidf",
}

# Model names rotate quickly (checked against provider docs on 2026-11-14). Override without touching code:
#   GEMINI_MODEL / OPENAI_MODEL / OLLAMA_MODEL in .env, or --llm-model on the command line.
_DEFAULT_LLM_MODEL = {
    "gemini": "gemini-3.8-flash",
    "openai": "gpt-5.6-luna",
    "ollama": "llama3.1:8b",
    "extractive": "extractive-baseline",
}


def default_llm_model(provider: str) -> str:
    return os.environ.get(f"{provider.upper()}_MODEL") or _DEFAULT_LLM_MODEL[provider]


@dataclass
class Config:
    pdf_path: str = "data/intro-to-ml-book.pdf"
    index_dir: str = "index"

    # --- Part 1: preprocessing
    first_title: str = "Chapter 1"   # first PDF-outline entry to keep (skips front matter)
    last_title: str = "Index"        # first PDF-outline entry to drop (index + colophon)
    chunk_size: int = 500            # tokens
    chunk_overlap: int = 100         # tokens

    # --- Part 2: embeddings + vector DB
    embedding_backend: str = "st"    # st | openai | gemini | tfidf
    embedding_model: str = ""        # "" -> DEFAULT_EMBEDDING_MODEL[backend]
    store: str = "faiss"             # faiss | chroma   (tfidf always uses a sparse store)

    # --- Part 3/4: retrieval + generation
    top_k: int = 5
    min_score: float = -1.0          # -1 -> DEFAULT_MIN_SCORE[backend]
    llm_provider: str = "gemini"     # gemini | openai | ollama | extractive
    llm_model: str = ""              # "" -> DEFAULT_LLM_MODEL[provider]

    def resolved(self) -> "Config":
        c = Config(**asdict(self))
        if not c.embedding_model:
            c.embedding_model = DEFAULT_EMBEDDING_MODEL[c.embedding_backend]
        if c.min_score < 0:
            c.min_score = DEFAULT_MIN_SCORE[c.embedding_backend]
        if not c.llm_model:
            c.llm_model = default_llm_model(c.llm_provider)
        return c

    def path(self, p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else ROOT / q
