"""Part 2b - vector stores with a common interface: build / search / save / load.

search(query_vec, k) -> list[(row_index, similarity)]  (higher = more similar, cosine)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class FaissStore:
    """Exact (brute-force) inner-product index. Vectors are L2-normalised, so IP == cosine.

    For ~1-3k chunks a flat index answers in well under a millisecond; approximate indexes
    (IVF/HNSW) only pay off beyond roughly 100k vectors.  Falls back to NumPy if FAISS is absent.
    """
    name = "faiss"

    def __init__(self):
        self.index = None
        self.matrix = None  # numpy fallback

    def build(self, vectors: np.ndarray, chunks=None):
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        try:
            import faiss

            self.index = faiss.IndexFlatIP(vectors.shape[1])
            self.index.add(vectors)
        except ImportError:
            self.matrix = vectors
        return self

    def search(self, q: np.ndarray, k: int):
        q = np.ascontiguousarray(q.reshape(1, -1), dtype="float32")
        if self.index is not None:
            D, I = self.index.search(q, k)
            return [(int(i), float(d)) for i, d in zip(I[0], D[0]) if i >= 0]
        sims = (self.matrix @ q[0])
        top = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top]

    def save(self, directory: Path):
        directory = Path(directory)
        if self.index is not None:
            import faiss

            faiss.write_index(self.index, str(directory / "faiss.index"))
        else:
            np.save(directory / "vectors.npy", self.matrix)

    @classmethod
    def load(cls, directory: Path):
        directory = Path(directory)
        s = cls()
        fi = directory / "faiss.index"
        if fi.exists():
            import faiss

            s.index = faiss.read_index(str(fi))
        else:
            s.matrix = np.load(directory / "vectors.npy")
        return s


class SparseStore:
    """Sparse TF-IDF matrix (rows are L2-normalised, so dot product == cosine)."""
    name = "sparse"

    def __init__(self):
        self.matrix = None

    def build(self, vectors, chunks=None):
        self.matrix = vectors.tocsr()
        return self

    def search(self, q, k: int):
        sims = (self.matrix @ q.T).toarray().ravel()
        top = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top]

    def save(self, directory: Path):
        import scipy.sparse as sp

        sp.save_npz(Path(directory) / "tfidf_matrix.npz", self.matrix)

    @classmethod
    def load(cls, directory: Path):
        import scipy.sparse as sp

        s = cls()
        s.matrix = sp.load_npz(Path(directory) / "tfidf_matrix.npz").tocsr()
        return s


class ChromaStore:
    """ChromaDB persistent collection (cosine space). Embeddings are supplied by us."""
    name = "chroma"

    def __init__(self, directory: Path | None = None):
        self.directory = directory
        self.col = None

    def _client(self, directory: Path):
        import chromadb

        return chromadb.PersistentClient(path=str(Path(directory) / "chroma"))

    def build(self, vectors: np.ndarray, chunks=None, directory: Path | None = None):
        client = self._client(directory or self.directory)
        try:
            client.delete_collection("ml_book")
        except Exception:
            pass
        self.col = client.create_collection("ml_book", metadata={"hnsw:space": "cosine"})
        for i in range(0, len(vectors), 500):
            sl = slice(i, i + 500)
            self.col.add(
                ids=[str(c.chunk_id) for c in chunks[sl]],
                embeddings=vectors[sl].tolist(),
                documents=[c.text for c in chunks[sl]],
                metadatas=[{"chapter": c.chapter, "section": c.section, "page_start": c.page_start}
                           for c in chunks[sl]])
        return self

    def search(self, q: np.ndarray, k: int):
        r = self.col.query(query_embeddings=[q.tolist()], n_results=k)
        return [(int(i), 1.0 - float(d)) for i, d in zip(r["ids"][0], r["distances"][0])]

    def save(self, directory: Path):  # Chroma persists on its own
        pass

    @classmethod
    def load(cls, directory: Path):
        s = cls(directory)
        s.col = s._client(directory).get_collection("ml_book")
        return s


def make_store(name: str, sparse: bool, directory: Path | None = None):
    if sparse:
        return SparseStore()
    if name == "chroma":
        return ChromaStore(directory)
    return FaissStore()


def load_store(name: str, directory: Path):
    return {"faiss": FaissStore, "sparse": SparseStore, "chroma": ChromaStore}[name].load(directory)
