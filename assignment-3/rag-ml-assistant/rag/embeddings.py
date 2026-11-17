"""Part 2a - embedding backends.

st      SentenceTransformers (local, free)     <- default: BAAI/bge-small-en-v1.5
openai  OpenAI embeddings API                  (needs OPENAI_API_KEY)
gemini  Google Gemini embeddings API           (needs GEMINI_API_KEY / GOOGLE_API_KEY)
tfidf   scikit-learn TF-IDF (sparse, offline)  <- zero-dependency baseline / fallback
All dense backends return L2-normalised float32 vectors, so inner product == cosine similarity.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np


def _l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float32")
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


class BaseEmbedder:
    kind = "dense"          # dense | sparse
    backend = ""
    model_name = ""

    def encode_docs(self, texts: list[str]):
        raise NotImplementedError

    def encode_query(self, text: str):
        raise NotImplementedError

    def save(self, directory: Path) -> None:  # only tf-idf needs state on disk
        pass


# ----------------------------------------------------------------- SentenceTransformers
_QUERY_PREFIX = {
    "bge": "Represent this sentence for searching relevant passages: ",
    "e5": "query: ",
}
_DOC_PREFIX = {"e5": "passage: "}


class SentenceTransformerEmbedder(BaseEmbedder):
    backend = "st"

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        low = model_name.lower()
        self.q_prefix = next((v for k, v in _QUERY_PREFIX.items() if k in low), "")
        self.d_prefix = next((v for k, v in _DOC_PREFIX.items() if k in low), "")
        self.max_seq_length = self.model.max_seq_length

    def encode_docs(self, texts):
        v = self.model.encode([self.d_prefix + t for t in texts], batch_size=32,
                              show_progress_bar=True, convert_to_numpy=True)
        return _l2(v)

    def encode_query(self, text):
        return _l2(self.model.encode([self.q_prefix + text], convert_to_numpy=True))[0]


# ----------------------------------------------------------------------------- OpenAI
class OpenAIEmbedder(BaseEmbedder):
    backend = "openai"

    def __init__(self, model_name: str):
        from openai import OpenAI

        self.model_name = model_name
        self.client = OpenAI()

    def _embed(self, batch):
        for attempt in range(5):
            try:
                r = self.client.embeddings.create(model=self.model_name, input=batch)
                return [d.embedding for d in r.data]
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)

    def encode_docs(self, texts):
        out = []
        for i in range(0, len(texts), 64):
            out.extend(self._embed(texts[i:i + 64]))
        return _l2(np.array(out))

    def encode_query(self, text):
        return _l2(np.array(self._embed([text])))[0]


# ----------------------------------------------------------------------------- Gemini
class GeminiEmbedder(BaseEmbedder):
    backend = "gemini"

    def __init__(self, model_name: str):
        from google import genai

        self.model_name = model_name
        self.client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY

    def _embed(self, batch, task):
        from google.genai import types

        for attempt in range(6):
            try:
                r = self.client.models.embed_content(
                    model=self.model_name, contents=batch,
                    config=types.EmbedContentConfig(task_type=task))
                return [e.values for e in r.embeddings]
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)

    def encode_docs(self, texts):
        out = []
        for i in range(0, len(texts), 50):
            out.extend(self._embed(texts[i:i + 50], "RETRIEVAL_DOCUMENT"))
            time.sleep(0.5)
        return _l2(np.array(out))

    def encode_query(self, text):
        return _l2(np.array(self._embed([text], "RETRIEVAL_QUERY")))[0]


# ----------------------------------------------------------------------------- TF-IDF
class TfidfEmbedder(BaseEmbedder):
    """Sparse lexical baseline. Needs no model download and no network."""
    kind = "sparse"
    backend = "tfidf"
    model_name = "tfidf"

    def __init__(self, vectorizer=None):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = vectorizer or TfidfVectorizer(
            sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_df=0.5,
            stop_words="english", lowercase=True, max_features=200_000, norm="l2")

    def encode_docs(self, texts):
        if len(texts) < 50:  # tiny corpora: frequency cut-offs would remove every term
            self.vec.set_params(min_df=1, max_df=1.0)
        return self.vec.fit_transform(texts)

    def encode_query(self, text):
        return self.vec.transform([text])

    def save(self, directory: Path) -> None:
        with open(Path(directory) / "tfidf.pkl", "wb") as f:
            pickle.dump(self.vec, f)

    @classmethod
    def load(cls, directory: Path) -> "TfidfEmbedder":
        with open(Path(directory) / "tfidf.pkl", "rb") as f:
            return cls(pickle.load(f))


def make_embedder(backend: str, model_name: str) -> BaseEmbedder:
    if backend == "st":
        return SentenceTransformerEmbedder(model_name)
    if backend == "openai":
        return OpenAIEmbedder(model_name)
    if backend == "gemini":
        return GeminiEmbedder(model_name)
    if backend == "tfidf":
        return TfidfEmbedder()
    raise ValueError(f"unknown embedding backend: {backend}")


def load_embedder(backend: str, model_name: str, directory: Path) -> BaseEmbedder:
    if backend == "tfidf":
        return TfidfEmbedder.load(directory)
    return make_embedder(backend, model_name)
