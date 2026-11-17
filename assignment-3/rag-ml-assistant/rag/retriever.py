"""Part 2c + Part 3 - build/persist the index and run query -> embedding -> top-k retrieval."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from .chunker import Chunk
from .config import Config
from .embeddings import load_embedder, make_embedder
from .vectorstore import load_store, make_store


@dataclass
class Retrieved:
    rank: int
    score: float
    chunk: Chunk


class RagIndex:
    def __init__(self, chunks: list[Chunk], embedder, store, manifest: dict):
        self.chunks = chunks
        self.embedder = embedder
        self.store = store
        self.manifest = manifest

    # ------------------------------------------------------------------ build / persist
    @classmethod
    def build(cls, chunks: list[Chunk], cfg: Config, directory: Path | None = None) -> "RagIndex":
        cfg = cfg.resolved()
        directory = Path(directory) if directory else cfg.path(cfg.index_dir)
        directory.mkdir(parents=True, exist_ok=True)
        embedder = make_embedder(cfg.embedding_backend, cfg.embedding_model)
        t0 = time.time()
        vectors = embedder.encode_docs([c.embed_text() for c in chunks])
        store = make_store(cfg.store, embedder.kind == "sparse", directory)
        store.build(vectors, chunks=chunks) if store.name != "chroma" else store.build(vectors, chunks, directory)
        manifest = {
            "embedding_backend": cfg.embedding_backend, "embedding_model": cfg.embedding_model,
            "store": store.name, "n_chunks": len(chunks),
            "dim": int(vectors.shape[1]), "chunk_size": cfg.chunk_size, "chunk_overlap": cfg.chunk_overlap,
            "embed_seconds": round(time.time() - t0, 2), "min_score": cfg.min_score,
            "pdf": Path(cfg.pdf_path).name,
        }
        idx = cls(chunks, embedder, store, manifest)
        idx.save(directory)
        return idx

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        self.store.save(directory)
        self.embedder.save(directory)
        (directory / "manifest.json").write_text(json.dumps(self.manifest, indent=2))

    @classmethod
    def load(cls, directory: Path) -> "RagIndex":
        directory = Path(directory)
        if not (directory / "manifest.json").exists():
            raise FileNotFoundError(
                f"No index found in {directory}. Build it first:  python -m rag.cli build")
        manifest = json.loads((directory / "manifest.json").read_text())
        chunks = [Chunk(**json.loads(l)) for l in open(directory / "chunks.jsonl", encoding="utf-8")]
        embedder = load_embedder(manifest["embedding_backend"], manifest["embedding_model"], directory)
        store = load_store(manifest["store"], directory)
        return cls(chunks, embedder, store, manifest)

    # ------------------------------------------------------------------------- search
    def search(self, query: str, k: int = 5) -> list[Retrieved]:
        """query -> embedding -> top-k most similar chunks (cosine similarity)."""
        qv = self.embedder.encode_query(query)
        hits = self.store.search(qv, k)
        return [Retrieved(rank=r, score=s, chunk=self.chunks[i]) for r, (i, s) in enumerate(hits, 1)]
