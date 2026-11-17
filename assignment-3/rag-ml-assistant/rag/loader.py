"""Part 1a - load the PDF, extract text from every page, clean it, explore it."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

# Page footers in this book look like "86 | Chapter 2: Supervised Learning" (even pages)
# or "Preface | xi" / "Supervised Machine Learning Algorithms | 47" (odd pages).
_FOOTER_A = re.compile(r"^\s*(?P<label>\d+|[ivxlc]+)\s*\|\s*(?P<title>\S.*)$", re.I)
_FOOTER_B = re.compile(r"^\s*(?P<title>\S.{2,}?)\s*\|\s*(?P<label>\d+|[ivxlc]+)\s*$", re.I)
_CODE_MARK = re.compile(r"^(In|Out)\[\d+\]:\s*$")
_TERMINAL = (".", "!", "?", ":")
_KERN_FIX = re.compile(r"\bY (?=(?:ou|our|ork)\b)")
HYPHEN = "\u2010"  # the book uses U+2010 for line-break hyphenation


@dataclass
class Page:
    pdf_page: int            # 1-based index in the PDF
    label: str               # printed page label (e.g. "47", "xi")
    chapter: str
    section: str
    raw_text: str
    text: str                # cleaned text (paragraphs separated by blank lines)
    in_scope: bool           # False for front matter / index
    footer_removed: bool


# --------------------------------------------------------------------------- outline
def _flatten_outline(reader: PdfReader) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []

    def walk(items, depth):
        for it in items:
            if isinstance(it, list):
                walk(it, depth + 1)
            else:
                try:
                    pg = reader.get_destination_page_number(it) + 1
                except Exception:
                    continue
                out.append((pg, depth, str(it.title).strip()))

    walk(reader.outline, 0)
    return out


def _page_titles(outline, n_pages):
    """For every page, the chapter (outline depth 0) and section (depth 1-2) it falls in."""
    by_page: dict[int, list[tuple[int, str]]] = {}
    for pg, depth, title in outline:
        by_page.setdefault(pg, []).append((depth, title))
    cur = {0: "Front matter", 1: "", 2: ""}
    res = {}
    for p in range(1, n_pages + 1):
        for depth, title in by_page.get(p, []):
            if depth > 2:
                continue
            cur[depth] = title
            for d in range(depth + 1, 3):
                cur[d] = ""
        section = cur[2] or cur[1]
        res[p] = (cur[0], section)
    return res


def _scope_range(outline, n_pages, first_title, last_title):
    first = next((pg for pg, d, t in outline if d == 0 and t.startswith(first_title)), 1)
    last = next((pg for pg, d, t in outline if d == 0 and t.startswith(last_title)), n_pages + 1)
    return first, last - 1


# --------------------------------------------------------------------------- cleaning
def _strip_footer(lines: list[str], known_titles: tuple[str, ...] = ()) -> tuple[list[str], str | None]:
    """Remove the running footer (checked on first/last two non-empty lines only)."""
    idx = [i for i, l in enumerate(lines) if l.strip()]
    if not idx:
        return lines, None

    # 1) footer glued onto the end of a code line, e.g. '...loc="best")Supervised Machine Learning Algorithms | 43'
    #    (tried first so that a short glued code line is not swallowed whole by the generic patterns below)
    last_i = idx[-1]
    last = lines[last_i]
    for t in sorted(known_titles, key=len, reverse=True):
        ma = re.search(rf"(?P<label>\d+)\s*\|\s*{re.escape(t)}\s*$", last)           # '...()150 | Chapter 3: Title'
        mb = re.search(rf"{re.escape(t)}\s*\|\s*(?P<label>\d+|[ivxlc]+)\s*$", last, re.I)  # '...()Title | 43'
        m = ma or mb
        if m:
            head = last[:m.start()]
            new_lines = lines[:last_i] + ([head] if head.strip() else []) + lines[last_i + 1:]
            return new_lines, m.group("label")

    # 2) footer on its own line
    for i in idx[-2:] + idx[:2]:
        for rx in (_FOOTER_A, _FOOTER_B):
            m = rx.match(lines[i])
            if m and len(lines[i]) < 90:
                return lines[:i] + lines[i + 1:], m.group("label")
    return lines, None


def _looks_like_prose(line: str) -> bool:
    w = line.split()
    return (len(w) >= 7 and line[:1].isupper() and not re.search(r"[={}\\]", line))


_CODE_START = re.compile(r"^[a-z_][\w\.]*(\(|\s=\s)")


def _looks_like_code_start(raw: str) -> bool:
    """Page begins in the middle of a code cell (no 'In[n]:' marker on this page)."""
    s = raw.strip()
    return s.startswith("#") or raw[:1] in (" ", "\t") or bool(_CODE_START.match(s))


def _reflow(lines: list[str]) -> str:
    """Join wrapped prose lines into paragraphs; keep code cells line-by-line."""
    out: list[str] = []
    para: list[str] = []
    code: list[str] = []
    in_code = False
    first = True

    def flush_para():
        if para:
            out.append(" ".join(para))
            para.clear()

    def flush_code():
        if code:
            out.append("\n".join(code))
            code.clear()

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            continue
        if first:
            first = False
            if _looks_like_code_start(line):
                in_code = True
        if _CODE_MARK.match(line.strip()):
            flush_para()
            in_code = True
            code.append(line.strip())
            continue
        if in_code:
            if _looks_like_prose(line.strip()):
                flush_code()
                in_code = False
            else:
                code.append(line)
                continue
        s = line.strip()
        if para:
            prev = para[-1]
            heading = (len(s.split()) <= 10 and s[:1].isupper()
                       and not s.endswith((".", ",", ";", ":", "?", "!", HYPHEN, "-"))
                       and prev.endswith(_TERMINAL))
            short_end = prev.endswith(_TERMINAL) and len(prev) < 70
            starts_fig = re.match(r"^(Figure|Table)\s+\d+-\d+\.", s)
            after_caption = (re.match(r"^(Figure|Table)\s+\d+-\d+\.", para[0]) is not None
                             and len(para) <= 3 and s[:1].isupper())
            prev_heading = len(para) == 1 and len(prev.split()) <= 10 and not prev.endswith(_TERMINAL + (",",))
            if heading or short_end or starts_fig or after_caption or (prev_heading and s[:1].isupper() and not prev.endswith(HYPHEN)):
                flush_para()
        para.append(s)
    flush_para()
    flush_code()
    return "\n\n".join(out)


def clean_page_text(raw: str, known_titles: tuple[str, ...] = ()) -> tuple[str, str | None, bool]:
    """Return (clean_text, printed_page_label, footer_removed)."""
    lines = raw.replace("\x00", "").split("\n")
    # NB: the footer must be removed BEFORE de-hyphenating, otherwise a page ending in a
    # hyphenated word would glue its footer ("param-\n118 | Chapter 2") onto the body text.
    lines, label = _strip_footer(lines, known_titles)
    if lines and lines[0].strip().upper().startswith("CHAPTER ") and lines[-1].strip().isdigit():
        lines = lines[:-1]                              # chapter-opener pages: stray chapter number
    t = "\n".join(lines)
    # re-join words hyphenated across line breaks, then normalise remaining U+2010
    t = re.sub(rf"(\w){HYPHEN}\n\s*(\w)", r"\1\2", t)
    t = t.replace(HYPHEN, "-").replace("\u00ad", "")
    t = unicodedata.normalize("NFKC", t)               # fi/fl ligatures, odd spaces, etc.
    t = re.sub(r"(?<=\w)\u2013(?=\w)", "-", t)         # "tf-idf" written with an en dash
    t = _KERN_FIX.sub("Y", t)                          # "Y ou" -> "You" (kerning artefact)
    t = t.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = _reflow(t.split("\n"))
    return text.strip(), label, label is not None


# --------------------------------------------------------------------------- public API
def load_document(pdf_path: str | Path, first_title="Chapter 1", last_title="Index") -> list[Page]:
    reader = PdfReader(str(pdf_path))
    n = len(reader.pages)
    outline = _flatten_outline(reader)
    titles = _page_titles(outline, n)
    first, last = _scope_range(outline, n, first_title, last_title)
    known = tuple({re.sub(r"^(Chapter \d+)\.", r"\1:", t).replace("\u2013", "-") for _, d, t in outline if d <= 1}
                  | {t.replace("\u2013", "-") for _, _, t in outline})

    pages: list[Page] = []
    for i, pg in enumerate(reader.pages, start=1):
        try:
            raw = pg.extract_text() or ""
        except Exception:
            raw = ""
        text, label, removed = clean_page_text(raw, known)
        chapter, section = titles[i]
        pages.append(Page(
            pdf_page=i, label=label or str(i), chapter=chapter, section=section,
            raw_text=raw, text=text, in_scope=first <= i <= last, footer_removed=removed))
    # pages without a footer (chapter openers, ...): infer the printed number from the dominant offset
    offs = Counter(p.pdf_page - int(p.label) for p in pages if p.footer_removed and p.label.isdigit())
    if offs:
        off = offs.most_common(1)[0][0]
        for p in pages:
            if not p.footer_removed and p.pdf_page > off:
                p.label = str(p.pdf_page - off)
    return pages


def explore(pages: list[Page]) -> dict:
    """Part 1 - document structure + text-quality diagnostics."""
    raw_all = "\n".join(p.raw_text for p in pages)
    scope = [p for p in pages if p.in_scope]
    clean_all = "\n\n".join(p.text for p in scope)
    chars_per_page = [len(p.raw_text) for p in pages]
    chapters = Counter(p.chapter for p in scope)
    code_lines = sum(1 for p in scope for l in p.raw_text.split("\n") if _CODE_MARK.match(l.strip()))
    stats = {
        "n_pages_total": len(pages),
        "n_pages_in_scope": len(scope),
        "scope_pdf_pages": [scope[0].pdf_page, scope[-1].pdf_page] if scope else None,
        "pages_excluded_front_matter": sum(1 for p in pages if not p.in_scope and scope and p.pdf_page < scope[0].pdf_page),
        "pages_excluded_index_and_back": sum(1 for p in pages if not p.in_scope and scope and p.pdf_page > scope[-1].pdf_page),
        "chars_raw_total": len(raw_all),
        "chars_clean_in_scope": len(clean_all),
        "words_clean_in_scope": len(clean_all.split()),
        "avg_chars_per_page_raw": round(sum(chars_per_page) / max(1, len(chars_per_page))),
        "empty_pages": [p.pdf_page for p in pages if len(p.raw_text.strip()) == 0],
        "near_empty_pages_lt_200_chars": [p.pdf_page for p in scope if len(p.raw_text.strip()) < 200],
        "footers_removed_pages": sum(1 for p in scope if p.footer_removed),
        "linebreak_hyphenations_fixed": len(re.findall(rf"\w{HYPHEN}\n\s*\w", raw_all)),
        "u2010_hyphens_total": raw_all.count(HYPHEN),
        "kerning_splits_Y_ou_fixed": len(_KERN_FIX.findall(raw_all)),
        "ligature_chars_normalised": len(re.findall("[\ufb00-\ufb06]", raw_all)),
        "code_cells_In_Out": code_lines,
        "non_ascii_chars_after_cleaning": sum(1 for c in clean_all if ord(c) > 127),
        "chapters": dict(chapters),
        "has_text_layer": any(len(p.raw_text.strip()) > 0 for p in pages),
    }
    return stats
