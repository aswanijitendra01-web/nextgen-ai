"""Part 1b - split the cleaned text into overlapping, token-sized chunks with metadata."""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

from .loader import Page

# --- token counting -----------------------------------------------------------------
# Uses tiktoken (cl100k_base) when available. Otherwise falls back to a regex tokenizer
# (words + punctuation), which tracks real BPE token counts within ~15-20% for English/code.
_ENC = None
try:  # pragma: no cover - depends on environment
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
    _ENC.encode("test")
except Exception:
    _ENC = None

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
TOKENIZER_NAME = "tiktoken/cl100k_base" if _ENC else "regex-approx (words+punctuation)"


def count_tokens(text: str) -> int:
    if _ENC is not None:
        return len(_ENC.encode(text, disallowed_special=()))
    return len(_TOKEN_RE.findall(text))


# --- data structures ----------------------------------------------------------------
@dataclass
class Unit:
    text: str
    tokens: int
    pdf_page: int
    label: str
    section: str
    sep: str  # separator to put *before* this unit when re-joining


@dataclass
class Chunk:
    chunk_id: int
    chapter: str
    section: str
    page_start: str      # printed page label
    page_end: str
    pdf_page_start: int
    pdf_page_end: int
    n_tokens: int
    text: str

    def embed_text(self) -> str:
        """Text that is embedded: chunk + a short breadcrumb that gives topical context."""
        crumb = self.chapter if not self.section else f"{self.chapter} > {self.section}"
        return f"{crumb}\n{self.text}"

    def citation(self) -> str:
        pages = self.page_start if self.page_start == self.page_end else f"{self.page_start}-{self.page_end}"
        sec = f", {self.section}" if self.section else ""
        return f"{self.chapter}{sec} (p. {pages})"

    def to_dict(self) -> dict:
        return asdict(self)


# --- unit building ------------------------------------------------------------------
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def _is_code(par: str) -> bool:
    return par.lstrip().startswith(("In[", "Out[")) or par.count("\n") >= 2


def _split_long(text: str, size: int, by: str) -> list[str]:
    """Split an oversized block (long code cell / giant sentence) into <= size pieces."""
    parts = text.split("\n") if by == "line" else text.split(" ")
    glue = "\n" if by == "line" else " "
    out, cur, cur_t = [], [], 0
    for p in parts:
        t = count_tokens(p) + 1
        if cur and cur_t + t > size:
            out.append(glue.join(cur))
            cur, cur_t = [], 0
        cur.append(p)
        cur_t += t
    if cur:
        out.append(glue.join(cur))
    return out


def _page_units(page: Page, size: int) -> list[Unit]:
    units: list[Unit] = []
    for par in [p for p in page.text.split("\n\n") if p.strip()]:
        if _is_code(par):
            pieces = [par] if count_tokens(par) <= size else _split_long(par, size, "line")
        else:
            pieces = []
            for s in _SENT_SPLIT.split(par):
                pieces.extend([s] if count_tokens(s) <= size else _split_long(s, size, "word"))
        for i, piece in enumerate(pieces):
            units.append(Unit(piece, count_tokens(piece), page.pdf_page, page.label, page.section,
                              "\n\n" if i == 0 else (" " if not _is_code(par) else "\n")))
    return units


# --- chunking -----------------------------------------------------------------------
def _window(units: list[Unit], size: int, overlap: int) -> list[list[Unit]]:
    windows, i, n = [], 0, len(units)
    while i < n:
        j, tok = i, 0
        while j < n and (j == i or tok + units[j].tokens <= size):
            tok += units[j].tokens
            j += 1
        windows.append(units[i:j])
        if j >= n:
            break
        k, ov = j, 0
        while k > i + 1 and ov + units[k - 1].tokens <= overlap:
            ov += units[k - 1].tokens
            k -= 1
        i = k
    return windows


def chunk_pages(pages: list[Page], chunk_size: int = 500, chunk_overlap: int = 100,
                min_tokens: int = 30) -> list[Chunk]:
    """Chunk the in-scope pages. Chunks never cross chapter boundaries but do cross pages."""
    assert 0 <= chunk_overlap < chunk_size, "overlap must be smaller than chunk size"
    scope = [p for p in pages if p.in_scope]
    chunks: list[Chunk] = []
    for chapter in dict.fromkeys(p.chapter for p in scope):          # ordered unique
        units: list[Unit] = []
        for p in (q for q in scope if q.chapter == chapter):
            units.extend(_page_units(p, chunk_size))
        for w in _window(units, chunk_size, chunk_overlap):
            text = w[0].text + "".join(u.sep + u.text for u in w[1:])
            n_tok = count_tokens(text)
            if n_tok < min_tokens:
                continue
            chunks.append(Chunk(
                chunk_id=len(chunks), chapter=chapter, section=w[0].section,
                page_start=w[0].label, page_end=w[-1].label, pdf_page_start=w[0].pdf_page,
                pdf_page_end=w[-1].pdf_page, n_tokens=n_tok, text=text))
    return chunks


def chunk_stats(chunks: list[Chunk]) -> dict:
    toks = sorted(c.n_tokens for c in chunks)
    n = len(toks)
    return {
        "n_chunks": n, "tokenizer": TOKENIZER_NAME,
        "tokens_min": toks[0], "tokens_median": toks[n // 2], "tokens_mean": round(sum(toks) / n, 1),
        "tokens_p95": toks[int(n * 0.95)], "tokens_max": toks[-1], "tokens_total": sum(toks),
    }
