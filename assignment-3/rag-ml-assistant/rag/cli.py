"""Command line interface:  python -m rag.cli <command> [options]

  explore      Part 1  - load PDF, show structure + text-quality report
  build        Part 1/2 - clean, chunk, embed, store in the vector DB
  search       Part 2/3 - show top-k retrieved chunks for a query
  ask          Part 4  - retrieval + LLM answer for one question
  chat         Part 5  - interactive Q&A loop
  experiments  Part 3  - compare k = 3/5/7 and chunk sizes
  demo         Part 4/5 - run the sample questions, write results/demo_answers.md
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

from .config import ROOT, Config


def _cfg(a) -> Config:
    return Config(
        pdf_path=a.pdf, index_dir=a.index_dir, chunk_size=a.chunk_size, chunk_overlap=a.chunk_overlap,
        embedding_backend=a.backend, embedding_model=a.embedding_model, store=a.store, top_k=a.k,
        min_score=a.min_score, llm_provider=a.provider, llm_model=a.llm_model).resolved()


def _print_hits(hits, width=100, chars=500):
    for h in hits:
        print(f"\n[{h.rank}] score={h.score:.3f}  {h.chunk.citation()}  (chunk {h.chunk.chunk_id}, {h.chunk.n_tokens} tok)")
        body = " ".join(h.chunk.text.split())[:chars] + ("..." if len(h.chunk.text) > chars else "")
        print(textwrap.fill(body, width, initial_indent="    ", subsequent_indent="    "))


def cmd_explore(a):
    from .chunker import chunk_pages, chunk_stats
    from .loader import explore, load_document
    cfg = _cfg(a)
    pages = load_document(cfg.path(cfg.pdf_path), cfg.first_title, cfg.last_title)
    stats = explore(pages)
    chunks = chunk_pages(pages, cfg.chunk_size, cfg.chunk_overlap)
    stats["chunking"] = {"chunk_size": cfg.chunk_size, "overlap": cfg.chunk_overlap, **chunk_stats(chunks)}
    print(json.dumps(stats, indent=2))
    sample = next(p for p in pages if p.pdf_page == 61)
    out = {"stats": stats,
           "sample_page_pdf61_raw": sample.raw_text[:1200], "sample_page_pdf61_clean": sample.text[:1200]}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "part1_exploration.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nsaved results/part1_exploration.json")


def cmd_build(a):
    from .pipeline import build_index
    build_index(_cfg(a))


def cmd_search(a):
    from .pipeline import RagPipeline
    p = RagPipeline.from_dir(_cfg(a))
    _print_hits(p.retrieve(a.query, a.k))


def cmd_ask(a):
    from .pipeline import RagPipeline
    p = RagPipeline.from_dir(_cfg(a))
    r = p.ask(a.query, a.k)
    print(f"\nQ: {r.question}\n\nA: {r.answer}\n")
    if r.refused:
        print(f"(refused: {r.reason or 'model reported no support in context'})")
    print("Sources:")
    for h in r.hits:
        print(f"  [{h.rank}] {h.score:.3f}  {h.chunk.citation()}")
    print(f"\n{r.seconds}s  provider={p.provider}  k={r.k}")


def cmd_chat(a):
    from .pipeline import RagPipeline
    p = RagPipeline.from_dir(_cfg(a))
    print(f"ML Knowledge Assistant  (provider={p.provider}, k={p.k}).  Type 'quit' to exit.")
    while True:
        try:
            q = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"quit", "exit", "q"}:
            break
        if not q:
            continue
        r = p.ask(q)
        print(f"\nAssistant> {r.answer}\n")
        print("  sources: " + " | ".join(f"[{h.rank}] {h.chunk.citation()} ({h.score:.2f})" for h in r.hits))


def cmd_experiments(a):
    from .experiments import evaluate_chunking, evaluate_k, load_questions, page_title_map, score_profile, to_markdown
    from .retriever import RagIndex
    cfg = _cfg(a)
    qs = load_questions()
    idx = RagIndex.load(cfg.path(cfg.index_dir))
    k_res = evaluate_k(idx, qs, page_title_map(cfg), (3, 5, 7))
    profile = score_profile(idx, qs)
    rows = None
    if not a.skip_chunking:
        rows = evaluate_chunking(cfg, qs, [(300, 50), (500, 100), (800, 150), (1000, 200)], 5)
    md = to_markdown(idx.manifest, k_res, rows, profile)
    print(md)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "experiments.md").write_text(md, encoding="utf-8")
    (ROOT / "results" / "experiments.json").write_text(
        json.dumps({"manifest": idx.manifest, "k": k_res, "chunking": rows, "score_profile": profile}, indent=2), encoding="utf-8")


def cmd_demo(a):
    from .experiments import load_questions
    from .generator import lexical_support
    from .pipeline import RagPipeline
    cfg = _cfg(a)
    p = RagPipeline.from_dir(cfg)
    qs = load_questions()
    items = [(q["q"], "in-scope") for q in qs["in_scope"] if q.get("demo")] + \
            [(q["q"], "OUT-OF-SCOPE (should refuse)") for q in qs["out_of_scope"] if q.get("demo")]
    L = [f"# Demo: sample queries and generated answers\n",
         f"- Embeddings: `{p.index.manifest['embedding_backend']}` / `{p.index.manifest['embedding_model']}`; "
         f"store: `{p.index.manifest['store']}`; chunks: {p.index.manifest['n_chunks']}",
         f"- Generator: `{p.provider}` / `{p.model}`; k={p.k}; refusal threshold={p.min_score}\n"]
    for q, kind in items:
        r = p.ask(q, p.k)
        sup = lexical_support(r.answer, r.hits) if not r.refused else None
        L += [f"## Q: {q}", f"*type: {kind}*\n", f"**Answer:** {r.answer}\n",
              f"- refused: **{r.refused}**" + (f" ({r.reason})" if r.reason else ""),
              f"- lexical support of answer in retrieved context: **{sup}**" if sup is not None else "- lexical support: n/a",
              f"- latency: {r.seconds}s", "- retrieved sources:"]
        L += [f"  - [{h.rank}] score {h.score:.3f} - {h.chunk.citation()}" for h in r.hits]
        if r.hits:
            ex = " ".join(r.hits[0].chunk.text.split())[:420]
            L += ["", f"> **Evidence (top chunk excerpt):** {ex}...", ""]
        print(f"done: {q[:60]}")
    out = ROOT / "results" / a.out
    out.write_text("\n".join(L), encoding="utf-8")
    print("saved", out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rag", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["explore", "build", "search", "ask", "chat", "experiments", "demo"])
    ap.add_argument("query", nargs="?", help="question / search text (search, ask)")
    d = Config()
    ap.add_argument("--pdf", default=d.pdf_path)
    ap.add_argument("--index-dir", default=d.index_dir)
    ap.add_argument("--chunk-size", type=int, default=d.chunk_size)
    ap.add_argument("--chunk-overlap", type=int, default=d.chunk_overlap)
    ap.add_argument("--backend", default=d.embedding_backend, choices=["st", "openai", "gemini", "tfidf"])
    ap.add_argument("--embedding-model", default="")
    ap.add_argument("--store", default=d.store, choices=["faiss", "chroma"])
    ap.add_argument("-k", type=int, default=d.top_k)
    ap.add_argument("--provider", default=d.llm_provider, choices=["gemini", "openai", "ollama", "extractive"])
    ap.add_argument("--llm-model", default="")
    ap.add_argument("--min-score", type=float, default=-1.0, help="refusal threshold (default: per-backend)")
    ap.add_argument("--skip-chunking", action="store_true", help="experiments: only run the k study")
    ap.add_argument("--out", default="demo_answers.md", help="demo: output file name inside results/")
    a = ap.parse_args(argv)
    if a.command in {"search", "ask"} and not a.query:
        ap.error(f"'{a.command}' needs a query")
    {"explore": cmd_explore, "build": cmd_build, "search": cmd_search, "ask": cmd_ask, "chat": cmd_chat,
     "experiments": cmd_experiments, "demo": cmd_demo}[a.command](a)


if __name__ == "__main__":
    main()
