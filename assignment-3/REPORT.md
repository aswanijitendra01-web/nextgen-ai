# RAG Machine Learning Knowledge Assistant — Assignment Report

**Dataset:** *Introduction to Machine Learning with Python* — A. Müller & S. Guido (O'Reilly, 2016), 392-page PDF.
**Stack:** pypdf · custom cleaner/chunker · SentenceTransformers `BAAI/bge-small-en-v1.5` · FAISS · Gemini (`gemini-3.8-flash`) / OpenAI / Ollama · Streamlit.

> ### Read this first: what was executed, and what you must still run
> The environment in which this project was written had **no internet access**, so the dense embedding model and the
> LLM could not be downloaded/called. To avoid shipping untested claims, the work was verified like this:
>
> | Component | Status |
> |---|---|
> | Part 1 – loading, cleaning, chunking, exploration | ✅ **Run on the real book**; all numbers below are real |
> | Retrieval pipeline, FAISS-fallback/sparse stores, index save/load, experiments, grounding gate, CLI, notebook | ✅ **Run on the real book** with the offline **TF-IDF** backend and the **extractive** generator (results below) |
> | 15 unit tests | ✅ pass |
> | `st` / `openai` / `gemini` embeddings, FAISS (native), ChromaDB, Gemini / OpenAI / Ollama generation, Streamlit UI | ⚠️ **Written but not executed here** (compile-checked only) |
>
> **Before you submit**, run these four commands with your API key set; they regenerate every number/answer in
> §2–§5 with the real models into `results/` (the offline numbers are kept in `results/offline_baseline/` for comparison):
> ```bash
> python -m rag.cli build && python -m rag.cli experiments && python -m rag.cli demo
> streamlit run app.py        # take a screenshot for the demonstration
> ```
> Then paste `results/experiments.md` and `results/demo_answers.md` into §3 and §4.2 (or link them). Calibrate `--min-score` as described in §4.4.

---

## 1. Part 1 — Data understanding & preprocessing

### 1.1 Loading and structure
* Loaded with **pypdf**; the PDF has a real text layer (no OCR needed) and a **bookmark outline**, which I use to attach `chapter` and `section` to every page.
* **392 pages** in total. In scope for retrieval: **PDF pages 15–380 (366 pages, ≈ 93.6 k words, ≈ 624 k characters)**, i.e. Chapter 1 to the Conclusion.
  Excluded: 14 front-matter pages (copyright, TOC, preface) and 12 back pages (Index, About the Authors, Colophon) — the index is a list of terms and page numbers that would only pollute retrieval.
* Printed page numbers differ from PDF page numbers (offset 14: roman-numeral front matter). Citations use the **printed** page, recovered from each page footer (verified: 0 mismatches over the 366 in-scope pages).

| Chapter | Pages |
|---|---|
| 1 Introduction | 24 |
| 2 Supervised Learning | 106 |
| 3 Unsupervised Learning and Preprocessing | 80 |
| 4 Representing Data and Engineering Features | 40 |
| 5 Model Evaluation and Improvement | 54 |
| 6 Algorithm Chains and Pipelines | 18 |
| 7 Working with Text Data | 34 |
| 8 Wrapping Up | 10 |

### 1.2 Text-quality issues found and how each is handled

| Issue (measured on the real PDF) | Effect if ignored | Fix |
|---|---|---|
| Running footer on 354 pages (`86 \| Chapter 2: Supervised Learning`, `Preface \| xi`) | Noise in every chunk; wrong page numbers | Detected on first/last lines, removed, page label kept |
| **736** words split across lines with U+2010 (`pow‐\nerful`) | Broken tokens (`pow`, `erful`) hurt both lexical and semantic matching | Re-joined (**after** footer removal — doing it before glued the footer onto the body; caught by a unit test) |
| Footer glued onto a code line (`…)Supervised Machine Learning Algorithms \| 43`) | Footer text inside code; lost page label | Matched against the PDF's own outline titles |
| Kerning artefact `Y ou` / `Y our` / `New Y ork` (110×) | Corrupts the most frequent word in prose | Targeted regex |
| `tf–idf` written with an en dash | Users type `tf-idf` → no match | Normalised to `-` inside words |
| **698** code cells (`In[n]:` / `Out[n]:`) | Re-flowing would destroy code | Detected; line structure preserved; code starting mid-page (no `In[` marker) also detected |
| Figure captions run into the next paragraph | Caption merged with body | Caption boundary heuristic |
| 5 blank pages, 8 pages < 200 chars (chapter openers, figure-only pages) | Empty/near-empty chunks | Dropped by a 30-token minimum chunk size |
| Footnotes extracted at the top of the following page text; identifiers broken without hyphen (`max_fea` / `tures`) | Minor noise | **Not fixed** (documented limitation) |

### 1.3 Chunking
* **Splitter:** custom, sentence-aware. Paragraphs → sentences; code cells are kept as indivisible units (split by line only if larger than the chunk size). Units are packed up to `chunk_size` tokens, and each new chunk re-starts from the trailing sentences of the previous one to give `chunk_overlap` tokens of overlap. **Chunks never cross chapter boundaries** but do cross pages.
* **Metadata per chunk:** chapter, section, printed page range, PDF page range, token count → used for citations and for evaluation.
* **Token counting:** `tiktoken` (`cl100k_base`) when installed, otherwise a regex approximation (words + punctuation). The numbers below use the approximation.
* **Chosen setting: 500 tokens / 100 overlap → 367 chunks** (mean 451, median 481, p95 499, max 500, min 48 tokens).
  Why not larger? The default embedding model has a 512-word-piece window; chunks of 800–1000 tokens would be silently truncated *in the embedding* (the LLM would still see the full text). The chunk-size experiment (§3.2) shows no clear retrieval gain from larger chunks, while they inflate the prompt 1.7–2.1×.
* Each chunk is embedded as `"<Chapter> > <Section>\n<text>"`: the breadcrumb gives code-heavy chunks topical context.

---

## 2. Part 2 — Embeddings & vector database

### 2.1 Embedding model
**`BAAI/bge-small-en-v1.5`** via SentenceTransformers (default; `--backend st`).
* 33 M parameters, **384-dimensional** vectors, 512-token input window; runs on CPU (typically a minute or two for this book); no API key or cost.
* Strong retrieval quality for its size on the MTEB benchmark, and it is trained for *asymmetric* search (short question → long passage). It expects an instruction prefix on **queries only** (`"Represent this sentence for searching relevant passages: "`) — implemented in `rag/embeddings.py`.
* Vectors are **L2-normalised**, so inner product = cosine similarity.
* Alternatives implemented behind the same interface: `openai` (`text-embedding-3-small`, 8k-token window — no truncation issue), `gemini` (`gemini-embedding-001`), and `tfidf` (sparse lexical baseline used for the offline verification).

### 2.2 Vector storage
* **FAISS `IndexFlatIP`** (exact inner-product search) over the normalised vectors. For ~370 vectors a brute-force search takes well under a millisecond and is *exact*; approximate indexes (IVF/HNSW) only pay off beyond ~10⁵ vectors, so they would add tuning complexity for no gain.
* **Persistence** (`index/`): `faiss.index` (vectors), `chunks.jsonl` (text + metadata; row *i* ↔ vector *i*), `manifest.json` (model name, dimension, chunk settings, refusal threshold). Loading rebuilds the embedder from the manifest, so an index can never be queried with a mismatched model.
* Optional **ChromaDB** store (`--store chroma`, cosine space, persistent). If FAISS is not installed a NumPy matrix product is used (the unit test compares search results with a brute-force NumPy ranking, and exercises FAISS itself whenever it is installed).
* TF-IDF baseline uses a sparse SciPy matrix (`SparseStore`).

### 2.3 Example retrieved chunks (real output, offline TF-IDF backend, k = 5)
*Query: "How does a random forest reduce overfitting?"*

| Rank | Score | Source |
|---|---|---|
| 1 | 0.172 | Ch. 2 Supervised Learning › Ensembles of Decision Trees (p. 83-84) |
| 2 | 0.143 | Ch. 2 › Ensembles of Decision Trees (p. 86-88) |
| 3 | 0.142 | Ch. 2 › Ensembles of Decision Trees (p. 85-86) |
| 4 | 0.142 | Ch. 2 › Ensembles of Decision Trees (p. 84-85) |
| 5 | 0.116 | Ch. 5 › Metrics for Binary Classification (p. 291-292) *(off-topic)* |

*Query: "What is principal component analysis (PCA)?"* → top-5 all from Ch. 3 › *Principal Component Analysis (PCA)* / *The Effect of Preprocessing on Supervised Learning* (scores 0.316 → 0.241).
Full tables with excerpts: `results/offline_baseline/demo_answers_offline_baseline.md`. *(Replace with the `st` + FAISS output from `python -m rag.cli search "…"`.)*

---

## 3. Part 3 — Retrieval pipeline & experiments

**Pipeline** (`RagIndex.search`): accept query → embed (with the model's query prefix) → similarity search → top-k `Retrieved(rank, score, chunk)`.

**Evaluation protocol.** 16 questions (`eval/questions.json`), each labelled with the **table-of-contents section(s) that contain the answer**. A retrieved chunk is *relevant* if it spans a page inside such a section. Metrics: **hit rate@k** (≥ 1 relevant chunk in top-k), **MRR@k**, **precision@k**, plus keyword coverage as a secondary signal. Labels are section-level (not passage-level) and the set is small, so differences of ±1 question (= 0.06) are not significant.

### 3.1 Effect of k (500/100 chunks) — *TF-IDF baseline, real run*
| k | hit rate | MRR | precision@k | keyword coverage | avg context tokens |
|---|---|---|---|---|---|
| 3 | 0.875 | 0.844 | 0.667 | 0.958 | 1 348 |
| 5 | 0.938 | 0.859 | 0.613 | 0.979 | 2 204 |
| 7 | 0.938 | 0.859 | 0.607 | 0.979 | 3 136 |

**Reading:** k = 3 → 5 recovers one more question (+0.06 hit rate); k = 7 adds **nothing** but costs 42 % more prompt tokens and slightly lowers precision (more noise for the LLM to ignore). **k = 5 is the default**; k = 3 is a reasonable low-latency/low-cost choice.
*(Re-run with `st`: `python -m rag.cli experiments`.)*

### 3.2 Effect of chunk size / overlap (k = 5) — *TF-IDF baseline, real run*
| chunk / overlap | #chunks | hit | MRR | precision@5 | context tokens |
|---|---|---|---|---|---|
| 300 / 50 | 612 | 0.875 | 0.833 | 0.637 | 1 296 |
| **500 / 100** | 367 | 0.938 | 0.859 | 0.613 | 2 204 |
| 800 / 150 | 219 | 0.938 | 0.906 | 0.588 | 3 803 |
| 1000 / 200 | 178 | 0.938 | 0.875 | 0.562 | 4 674 |

**Reading:** small chunks (300) lose context and miss a question; 800 has the best MRR but costs 1.7× the tokens; precision falls monotonically as chunks grow (each chunk spans more topics). 500/100 is the best trade-off and fits the embedding window. With the dense model the picture may differ — rerun.

---

## 4. Part 4 — Answer generation (RAG)

### 4.1 Prompt design (`rag/generator.py`)
**System prompt** (temperature 0 where the provider allows it; Gemini 3.x is left at its recommended default):
1. Answer using **only** the numbered CONTEXT excerpts; no outside knowledge "even if you are sure".
2. If the context lacks the answer, reply **exactly** `I couldn't find this in the book.` (a fixed string the app can detect).
3. If only partly answered: answer that part and state what is missing.
4. **Cite** supporting excerpts inline as `[1]`, `[2]`; never cite an unused one.
5. **Never invent** numbers, function names, parameters or code; quote identifiers exactly.
6. Concise, no "according to the context" preamble.

**User prompt:** `CONTEXT:` blocks `[n] Source: <chapter, section (p. x-y)>` + chunk text separated by `---`, then `QUESTION:`.
*Why this design:* numbered, source-tagged context makes every claim traceable; a fixed refusal string turns "don't know" into a checkable behaviour instead of a free-form apology; temperature 0 removes sampling variation; rule 5 targets the most damaging failure for a programming textbook (plausible but non-existent API calls).

### 4.2 Sample queries and answers
Full file: `results/demo_answers.md` (regenerate with `python -m rag.cli demo`). **The answers below are from the offline *extractive* generator** (no LLM): it returns verbatim sentences with citations, so it cannot hallucinate but is far less fluent than an LLM answer — it is a baseline and a safety net, not the intended final output.

**Q: How does a random forest reduce overfitting?**
> There are two ways in which the trees in a random forest are randomized: by selecting the data points used to build a tree and by selecting the features in each split test. [1] If we build many trees, all of which work well and overfit in different ways, we can reduce the amount of overfitting by averaging their results. [1] A random forest is essentially a collection of decision trees, where each tree is slightly different from the others. [1] …

**Q: What is the difference between precision and recall?**
> There is a trade-off between optimizing recall and optimizing precision. [1] … if you find a model that predicts only the single data point it is most sure about as positive and the rest as negative, then precision will be perfect (…), but recall will be very bad. [1]

**Q: Who won the 2018 FIFA World Cup?** (out of scope) → `I couldn't find this in the book.` (refused by the score gate: best score 0.090 < 0.100).

### 4.3 Evidence of grounding
1. **Source-tagged answers** — every sentence carries `[n]` pointing to a chunk with chapter/section/page; the app shows the passages in an expandable panel.
2. **Refusal when unsupported** — gate before the LLM + prompt rule 2 (see below for measured behaviour).
3. **`lexical_support`** metric — share of the answer's content words that occur in the retrieved context; a low value flags possible hallucination (unit-tested: a made-up answer about transformers scores < 0.3 against ML-book chunks). It is trivially 1.0 for the extractive baseline, so it is only informative for LLM answers.
4. **The exact prompt sent** is stored on every `Answer` (`r.prompt`) so any answer can be audited.

### 4.4 Measured weaknesses (honest assessment)
* **The TF-IDF score gate is unreliable** (measured on the 16 in-scope + 3 out-of-scope questions at threshold 0.10): it wrongly refused **2 / 16** answerable questions (*supervised vs. unsupervised*, *feature scaling*) and let **1 / 3** out-of-scope questions through. The "transformers / self-attention" question — plausible-sounding but not in a 2016 book — scored 0.127 and the extractive baseline then returned irrelevant sentences. Lexical scores simply don't separate "related" from "unrelated" well.
* **Therefore:** with the LLM generator, prompt rule 2 is the primary defence against such questions, and the score gate is only a cheap pre-filter. With dense embeddings the score ranges separate better, but **do not trust the default threshold** (`rag/config.py`): run `python -m rag.cli experiments`, read the *similarity-score profile* table (in-scope min vs out-of-scope max) and set `--min-score` between them. Keep the threshold conservative (a false refusal is worse than letting the LLM decide).
* Retrieval of definitional questions can favour a chunk that *uses* a term over one that *defines* it (e.g. "cross-validation" → nested-CV chunk first with TF-IDF). Dense retrieval should improve this; a cross-encoder re-ranker is the natural next step.
* The extractive baseline occasionally emits a code-comment or fragment as a "sentence".

---

## 5. Part 5 — End-to-end application

* **Streamlit app** (`streamlit run app.py`): chat UI; sidebar to choose generator, model, **k (1–10)**, refusal threshold, show/hide passages, and **(re)build the index from the PDF** (document processing → retrieval → generation in one place). Each answer has an expandable *Sources* panel with chapter/section/page, similarity score and the passage. Example-question buttons and a refusal notice for unsupported questions are included.
* **CLI:** `python -m rag.cli chat` (terminal Q&A loop), `ask`, `search`.
* **Notebook:** `notebooks/RAG_Assignment.ipynb` runs Parts 1–5 top to bottom (executed here offline with `BACKEND="tfidf"`, `PROVIDER="extractive"`: 15/15 code cells ran without error).
* **Demonstration of multiple queries:** `python -m rag.cli demo` runs 8 in-scope + 2 out-of-scope questions and writes retrieved sources, scores, answers, refusal flags and evidence excerpts to `results/demo_answers.md`. *(Add a Streamlit screenshot here after running it.)*

---

## 6. Limitations and future work
* Text only — the book's plots and diagrams are not indexed (could add captions/figure-aware parsing or a multimodal model).
* Small, section-level-labelled evaluation set; add passage-level labels and LLM-as-judge faithfulness scoring (e.g. RAGAS) for a stronger evaluation.
* Next steps: hybrid retrieval (BM25 + dense with reciprocal-rank fusion — the TF-IDF results show lexical matching is complementary), cross-encoder re-ranking, query rewriting for follow-up questions in chat, and fixing mid-identifier line breaks.
* The book is from 2016 (scikit-learn 0.18): it can't answer questions about transformers, LLMs, or newer APIs — by design the assistant should say so.

## 7. Reproducibility
`pip install -r requirements.txt` · book at `data/intro-to-ml-book.pdf` · `python -m rag.cli explore|build|experiments|demo` · `python -m pytest -q` (15 offline tests). All random-free: results are deterministic apart from the LLM provider.
