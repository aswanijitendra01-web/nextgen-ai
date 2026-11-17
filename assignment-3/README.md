# 📘 RAG Machine Learning Knowledge Assistant

An end-to-end **Retrieval-Augmented Generation** system that answers machine-learning questions **only from the
text of a textbook** – *Introduction to Machine Learning with Python* (Müller & Guido, 392 pages) – and cites the
chapter and page of every passage it used.

```
PDF ─► load & clean ─► chunk (500 tok / 100 overlap + chapter·section·page metadata)
                          │
                          ▼
              embed (bge-small-en-v1.5) ─► FAISS index (cosine)
                                                   ▲
 question ─► embed ─► top-k chunks ─► grounded prompt ─► LLM (Gemini / OpenAI / Ollama) ─► answer + [n] citations
                          └─► score gate: refuse if the best chunk is too dissimilar
```

| Assignment part | Where |
|---|---|
| 1 Data understanding & preprocessing | `rag/loader.py`, `rag/chunker.py`, `python -m rag.cli explore` |
| 2 Embeddings & vector DB | `rag/embeddings.py`, `rag/vectorstore.py`, `rag/retriever.py` |
| 3 Retrieval pipeline + k experiments | `rag/retriever.py`, `rag/experiments.py`, `eval/questions.json` |
| 4 Grounded answer generation | `rag/generator.py` |
| 5 End-to-end app | `app.py` (Streamlit), `rag/cli.py chat`, `notebooks/RAG_Assignment.ipynb` |
| Write-up of all deliverables | **[`REPORT.md`](REPORT.md)** |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                                      # add GEMINI_API_KEY (or OPENAI_API_KEY)
# put the book at data/intro-to-ml-book.pdf               (not committed - copyrighted)

python -m rag.cli explore                                 # Part 1: structure + text-quality report
python -m rag.cli build                                   # Parts 1-2: chunk, embed, store in FAISS  -> index/
python -m rag.cli search "What is overfitting?" -k 5      # Part 3: see retrieved chunks
python -m rag.cli ask "How does a random forest reduce overfitting?"   # Part 4
python -m rag.cli experiments                             # Part 3: k = 3/5/7 + chunk-size study -> results/
python -m rag.cli demo                                    # Part 4/5: sample Q&A -> results/demo_answers.md
streamlit run app.py                                      # Part 5: web app
python -m rag.cli chat                                    # Part 5: terminal chat
```

**No API key / no downloads?** Everything also runs fully offline with a TF-IDF baseline and an extractive
(copy-the-best-sentences) generator – lower quality, but a good smoke test:

```bash
python -m rag.cli build --backend tfidf --index-dir index_tfidf
python -m rag.cli ask "What is PCA?" --backend tfidf --index-dir index_tfidf --provider extractive
```

### Options

| Flag | Values | Default |
|---|---|---|
| `--backend` | `st` (SentenceTransformers), `openai`, `gemini`, `tfidf` | `st` → `BAAI/bge-small-en-v1.5` |
| `--store` | `faiss`, `chroma` | `faiss` |
| `--provider` | `gemini`, `openai`, `ollama` (local), `extractive` (offline) | `gemini` |
| `--chunk-size` / `--chunk-overlap` | tokens | `500` / `100` |
| `-k` | top-k chunks | `5` |
| `--min-score` | refusal threshold (cosine) | per backend – **calibrate with `experiments`** |

Model names are configurable (`--embedding-model`, `--llm-model`, or `GEMINI_MODEL` / `OPENAI_MODEL` / `OLLAMA_MODEL` in `.env`).
Defaults (`gemini-3.8-flash`, `gpt-5.6-luna`) were checked against provider docs on 2026-11-14 – provider model names rotate
quickly, so if you get a "model not found" error, pick a current model from the provider's list and set the env var.

## Design decisions in one minute

* **Clean before chunking.** Footers, line-break hyphenation, kerning artefacts and code cells are handled explicitly
  (details and numbers in `REPORT.md`). Chunks never cross chapter boundaries and carry `chapter · section · page` metadata.
* **Embed a breadcrumb + text** (`Chapter > Section` prefix) so short or code-heavy chunks still carry topical context.
* **Exact FAISS search** (`IndexFlatIP` on L2-normalised vectors = cosine). With ~370 vectors an approximate index would add complexity for no benefit.
* **Three layers of grounding:** score gate → strict system prompt (context-only, cite `[n]`, fixed refusal sentence, temperature 0) → numbered source-tagged context.
* **Honest evaluation:** retrieval is scored against *section-level labels from the PDF's own table of contents* (hit rate, MRR, precision@k).

## Known limitations

* Page footnotes are extracted at the top of the following text; identifiers broken across lines without a hyphen (e.g. `max_fea` / `tures`) cannot be repaired automatically.
* Figures/plots are not indexed (text only). The book dates from 2016: it does not cover transformers, modern deep learning or recent scikit-learn APIs – the assistant should say so rather than guess.
* The score gate is a coarse filter; its threshold depends on the embedding model and must be calibrated (see `REPORT.md` §4.4).
* Retrieval labels are section-level (not passage-level) and the eval set is small (16 questions).

## Tests

```bash
python -m pytest -q   # 15 offline unit tests: cleaning, chunking, vector stores, index round-trip, grounding gate
```

## Push to GitHub

```bash
git init && git add . && git commit -m "RAG ML knowledge assistant"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```
`.gitignore` already excludes `.env`, the generated `index*/` folders and the (copyrighted) book PDF.
