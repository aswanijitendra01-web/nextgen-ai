"""Part 3 experiments: effect of k (3/5/7) and of chunk size/overlap on retrieval quality.

Ground truth: every in-scope question is labelled with the TOC section(s) of the book that contain the answer
(eval/questions.json). A retrieved chunk is relevant if it spans a page inside one of those sections.
Metrics: hit rate@k, MRR@k, precision@k (+ keyword coverage as a secondary signal).
"""
from __future__ import annotations

import json
import shutil
import statistics as st
from dataclasses import replace
from pathlib import Path

from .chunker import chunk_pages, chunk_stats
from .config import ROOT, Config
from .loader import load_document
from .retriever import RagIndex


def load_questions(path: Path | None = None) -> dict:
    return json.loads((path or ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))


def _coverage(hits, kws) -> float:
    ctx = " ".join(h.chunk.text for h in hits).lower()
    return sum(k in ctx for k in kws) / len(kws)


def page_title_map(cfg: Config) -> dict[int, str]:
    """pdf page -> TOC section title (lower-cased, en-dash normalised); used as ground truth."""
    pages = load_document(cfg.path(cfg.pdf_path), cfg.first_title, cfg.last_title)
    return {p.pdf_page: p.section.lower().replace("\u2013", "-") for p in pages}


def _relevant(chunk, wanted: set[str], titles: dict[int, str]) -> bool:
    """A chunk is relevant if any page it spans lies in a TOC section that contains the answer."""
    return any(titles.get(pg, "") in wanted for pg in range(chunk.pdf_page_start, chunk.pdf_page_end + 1))


def evaluate_k(index: RagIndex, qs: dict, titles: dict[int, str], ks=(3, 5, 7)) -> dict:
    """hit@k      = share of questions with >=1 relevant chunk in the top-k
    MRR@k      = mean reciprocal rank of the first relevant chunk (0 if none in top-k)
    precision@k= mean share of the top-k chunks that are relevant
    keyword_coverage = share of expected keywords found in the concatenated top-k text"""
    res = {}
    kmax = max(ks)
    cache = [(q, index.search(q["q"], kmax)) for q in qs["in_scope"]]
    for k in ks:
        hit, rr, prec, covs, ctx = [], [], [], [], []
        for q, hits in cache:
            h = hits[:k]
            wanted = set(q["sections"])
            rel = [_relevant(x.chunk, wanted, titles) for x in h]
            hit.append(any(rel))
            rr.append(next((1 / i for i, r in enumerate(rel, 1) if r), 0.0))
            prec.append(sum(rel) / k)
            covs.append(_coverage(h, q["kw"]))
            ctx.append(sum(x.chunk.n_tokens for x in h))
        res[k] = {
            "hit_rate": round(st.mean(hit), 3), "mrr": round(st.mean(rr), 3),
            "precision": round(st.mean(prec), 3), "keyword_coverage": round(st.mean(covs), 3),
            "avg_context_tokens": round(st.mean(ctx)),
            "per_question_hit": {q["q"]: bool(h_) for (q, _), h_ in zip(cache, hit)},
        }
    return res


def score_profile(index: RagIndex, qs: dict) -> dict:
    """Top-1 similarity for in-scope vs out-of-scope questions -> helps pick `min_score`."""
    ins = [index.search(q["q"], 1)[0].score for q in qs["in_scope"]]
    oos = {q["q"]: index.search(q["q"], 1)[0].score for q in qs["out_of_scope"]}
    return {
        "in_scope_top1": {"min": round(min(ins), 3), "median": round(st.median(ins), 3), "max": round(max(ins), 3)},
        "out_of_scope_top1": {k: round(v, 3) for k, v in oos.items()},
        "suggested_min_score": round((max(oos.values()) + min(ins)) / 2, 3) if max(oos.values()) < min(ins) else None,
    }


def evaluate_chunking(cfg: Config, qs: dict, settings, k: int = 5) -> list[dict]:
    cfg = cfg.resolved()
    pages = load_document(cfg.path(cfg.pdf_path), cfg.first_title, cfg.last_title)
    titles = {p.pdf_page: p.section.lower().replace("\u2013", "-") for p in pages}
    rows = []
    for size, ov in settings:
        chunks = chunk_pages(pages, size, ov)
        tmp = ROOT / "results" / f"_tmp_{size}_{ov}"
        idx = RagIndex.build(chunks, replace(cfg, chunk_size=size, chunk_overlap=ov), tmp)
        ev = evaluate_k(idx, qs, titles, (k,))[k]
        rows.append({"chunk_size": size, "overlap": ov, "n_chunks": len(chunks),
                     "mean_chunk_tokens": chunk_stats(chunks)["tokens_mean"],
                     "hit_rate": ev["hit_rate"], "mrr": ev["mrr"], "precision": ev["precision"],
                     "keyword_coverage": ev["keyword_coverage"], "avg_context_tokens": ev["avg_context_tokens"]})
        shutil.rmtree(tmp, ignore_errors=True)
    return rows


def to_markdown(manifest: dict, k_res: dict, chunk_rows: list[dict] | None, profile: dict) -> str:
    L = [f"## Retrieval experiments\n",
         f"Embedding: `{manifest['embedding_backend']}` / `{manifest['embedding_model']}` · store: `{manifest['store']}` · "
         f"chunks: {manifest['n_chunks']} (size {manifest['chunk_size']}, overlap {manifest['chunk_overlap']})\n",
         "### Effect of k\n", "| k | hit rate | MRR | precision@k | keyword coverage | avg context tokens |", "|---|---|---|---|---|---|"]
    for k, r in k_res.items():
        L.append(f"| {k} | {r['hit_rate']} | {r['mrr']} | {r['precision']} | {r['keyword_coverage']} | {r['avg_context_tokens']} |")
    if chunk_rows:
        L += ["\n### Effect of chunk size / overlap (k=5)\n",
              "| chunk size | overlap | #chunks | mean tokens | hit rate | MRR | precision@5 | keyword coverage | avg context tokens |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in chunk_rows:
            L.append(f"| {r['chunk_size']} | {r['overlap']} | {r['n_chunks']} | {r['mean_chunk_tokens']} | "
                     f"{r['hit_rate']} | {r['mrr']} | {r['precision']} | {r['keyword_coverage']} | {r['avg_context_tokens']} |")
    L += ["\n### Similarity-score profile (for choosing the refusal threshold)\n",
          f"In-scope top-1 score: {profile['in_scope_top1']}", "",
          "| out-of-scope question | top-1 score |", "|---|---|"]
    L += [f"| {q} | {s} |" for q, s in profile["out_of_scope_top1"].items()]
    L.append(f"\nSuggested `min_score`: **{profile['suggested_min_score']}**")
    return "\n".join(L) + "\n"
