"""Offline unit tests (no model downloads / API keys needed):  pytest -q"""
import json
import numpy as np
import pytest

from rag.chunker import chunk_pages, count_tokens
from rag.config import Config
from rag.generator import NOT_FOUND, build_user_prompt, extractive_answer, generate, lexical_support
from rag.loader import Page, clean_page_text
from rag.retriever import RagIndex, Retrieved
from rag.vectorstore import FaissStore, SparseStore


def _page(i, text, chapter="Chapter 1. A", section="S"):
    return Page(i, str(i), chapter, section, text, text, True, True)


# ------------------------------------------------------------------ cleaning
def test_hyphenation_rejoined_and_footer_removed():
    raw = "Linear models can be very pow\u2010\nerful for many datasets.\n46 | Chapter 2: Supervised Learning"
    text, label, removed = clean_page_text(raw)
    assert "powerful" in text and "|" not in text and label == "46" and removed


def test_footer_not_glued_when_page_ends_in_hyphenated_word():
    raw = "some words that end with a param\u2010\n118 | Chapter 2: Supervised Learning"
    text, label, _ = clean_page_text(raw)
    assert label == "118" and "118" not in text


def test_glued_footer_after_code_line():
    raw = 'In[1]:\nplt.show()Supervised Machine Learning Algorithms | 43'
    text, label, _ = clean_page_text(raw, ("Supervised Machine Learning Algorithms",))
    assert label == "43" and text.endswith("plt.show()")


def test_kerning_and_en_dash_fixes():
    text, _, _ = clean_page_text("Y ou can use tf\u2013idf here.")
    assert "You can use tf-idf here." in text


def test_code_cells_keep_line_structure():
    raw = "Here is code:\nIn[1]:\nx = 1\ny = 2\nOut[1]:\n3\nThe result shows that the value is three as expected."
    text, _, _ = clean_page_text(raw)
    assert "x = 1\ny = 2" in text


# ------------------------------------------------------------------ chunking
def test_chunks_respect_size_overlap_and_chapter_boundaries():
    para = " ".join(f"Sentence number {i} talks about machine learning models." for i in range(400))
    pages = [_page(1, para, "Chapter 1. A"), _page(2, para, "Chapter 2. B")]
    chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=40)
    assert all(c.n_tokens <= 200 for c in chunks)
    assert {c.chapter for c in chunks} == {"Chapter 1. A", "Chapter 2. B"}
    same = [c for c in chunks if c.chapter == "Chapter 1. A"]
    # consecutive chunks share text (overlap)
    assert same[0].text.split(". ")[-2] in same[1].text


def test_chunking_makes_progress_with_giant_units():
    chunks = chunk_pages([_page(1, "word " * 3000)], chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 10 and all(c.n_tokens <= 100 for c in chunks)


def test_bad_overlap_rejected():
    with pytest.raises(AssertionError):
        chunk_pages([_page(1, "a b c")], 100, 100)


# ------------------------------------------------------------------ vector stores
def test_faiss_or_numpy_store_matches_bruteforce():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 32)).astype("float32")
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    store = FaissStore().build(X)
    q = X[17] + 0.01 * rng.normal(size=32).astype("float32")
    q /= np.linalg.norm(q)
    got = [i for i, _ in store.search(q, 5)]
    assert got[0] == 17 and got == list(np.argsort(-(X @ q))[:5])


def test_store_roundtrip(tmp_path):
    X = np.eye(4, dtype="float32")
    FaissStore().build(X).save(tmp_path)
    assert FaissStore.load(tmp_path).search(X[2], 1)[0][0] == 2


# ------------------------------------------------------------------ index end-to-end (tf-idf)
def _toy_chunks():
    pages = [_page(1, "Overfitting happens when a model is too complex and memorizes the training set. "
                      "It then generalizes badly to new data. " * 5, "Chapter 1. A", "Overfitting"),
             _page(2, "The k-means algorithm assigns points to the closest cluster center and updates the centers. " * 6,
                   "Chapter 2. B", "Clustering"),
             _page(3, "Cross-validation splits the data into folds and evaluates generalization on each fold. " * 6,
                   "Chapter 3. C", "Cross-validation")]
    return chunk_pages(pages, 120, 20, min_tokens=5)


def test_index_build_save_load_search(tmp_path):
    cfg = Config(embedding_backend="tfidf", chunk_size=120, chunk_overlap=20)
    idx = RagIndex.build(_toy_chunks(), cfg, tmp_path)
    idx2 = RagIndex.load(tmp_path)
    hit = idx2.search("how does k-means choose cluster centers", 2)[0]
    assert hit.chunk.section == "Clustering" and hit.score > 0
    assert json.loads((tmp_path / "manifest.json").read_text())["n_chunks"] == len(idx.chunks)


# ------------------------------------------------------------------ generation / grounding
def _hits(tmp_path):
    cfg = Config(embedding_backend="tfidf", chunk_size=120, chunk_overlap=20)
    return RagIndex.build(_toy_chunks(), cfg, tmp_path).search("what is overfitting", 3)


def test_gate_refuses_without_calling_llm(tmp_path):
    out = generate("anything", _hits(tmp_path), provider="openai", model="x", min_score=10.0)
    assert out["refused"] and out["answer"] == NOT_FOUND and out["prompt"] is None


def test_prompt_contains_numbered_sources_and_question(tmp_path):
    hits = _hits(tmp_path)
    prompt = build_user_prompt("what is overfitting", hits)
    assert "[1] Source:" in prompt and "QUESTION: what is overfitting" in prompt


def test_extractive_answer_is_verbatim_and_cited(tmp_path):
    hits = _hits(tmp_path)
    ans = extractive_answer("what is overfitting", hits)
    assert "[1]" in ans and lexical_support(ans, hits) == 1.0


def test_lexical_support_flags_unsupported_answer(tmp_path):
    assert lexical_support("Transformers use self-attention heads with positional encodings.", _hits(tmp_path)) < 0.3
