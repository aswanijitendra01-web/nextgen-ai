"""Part 5 - end-to-end pipeline: PDF -> chunks -> index  and  question -> retrieve -> generate."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from .chunker import chunk_pages, chunk_stats
from .config import Config
from .generator import generate
from .loader import explore, load_document
from .retriever import RagIndex, Retrieved


def build_index(cfg: Config, directory: Path | None = None, verbose: bool = True) -> RagIndex:
    cfg = cfg.resolved()
    pdf = cfg.path(cfg.pdf_path)
    if not pdf.exists():
        raise FileNotFoundError(f"PDF not found: {pdf}  (put the book at data/intro-to-ml-book.pdf)")
    pages = load_document(pdf, cfg.first_title, cfg.last_title)
    chunks = chunk_pages(pages, cfg.chunk_size, cfg.chunk_overlap)
    if verbose:
        print(f"pages={len(pages)}  in-scope={sum(p.in_scope for p in pages)}  chunks={len(chunks)}")
        print("chunk stats:", chunk_stats(chunks))
    idx = RagIndex.build(chunks, cfg, directory)
    if verbose:
        print("index built:", idx.manifest)
    return idx


@dataclass
class Answer:
    question: str
    answer: str
    refused: bool
    reason: str
    hits: list[Retrieved]
    k: int
    seconds: float
    prompt: str | None = None


class RagPipeline:
    def __init__(self, index: RagIndex, provider: str = "gemini", model: str = "",
                 min_score: float | None = None, k: int = 5):
        from .config import DEFAULT_MIN_SCORE, default_llm_model

        self.index = index
        self.provider = provider
        self.model = model or default_llm_model(provider)
        self.min_score = min_score if min_score is not None else index.manifest.get(
            "min_score", DEFAULT_MIN_SCORE[index.manifest["embedding_backend"]])
        self.k = k

    @classmethod
    def from_dir(cls, cfg: Config) -> "RagPipeline":
        cfg = cfg.resolved()
        idx = RagIndex.load(cfg.path(cfg.index_dir))
        ms = cfg.min_score if cfg.min_score >= 0 else None
        # the index may have been built with a different backend than cfg says -> trust the index
        from .config import DEFAULT_MIN_SCORE
        ms = ms if cfg.embedding_backend == idx.manifest["embedding_backend"] else DEFAULT_MIN_SCORE[idx.manifest["embedding_backend"]]
        return cls(idx, cfg.llm_provider, cfg.llm_model, ms, cfg.top_k)

    def retrieve(self, question: str, k: int | None = None) -> list[Retrieved]:
        return self.index.search(question, k or self.k)

    def ask(self, question: str, k: int | None = None) -> Answer:
        t0 = time.time()
        k = k or self.k
        hits = self.retrieve(question, k)
        out = generate(question, hits, self.provider, self.model, self.min_score)
        return Answer(question, out["answer"], out["refused"], out["reason"], hits, k,
                      round(time.time() - t0, 2), out["prompt"])
