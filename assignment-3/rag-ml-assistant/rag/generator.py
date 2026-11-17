"""Part 4 - grounded answer generation.

Providers:  gemini | openai | ollama (local) | extractive (offline, no LLM)
Grounding is enforced three ways:
  1. a retrieval-score gate - if the best chunk is not similar enough we refuse *without* calling the LLM
  2. a strict system prompt (answer only from CONTEXT, cite [n], refuse when unsupported; temperature 0 where the provider allows it)
  3. numbered, source-tagged context so every claim can be traced back to a chapter/page
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

from .retriever import Retrieved

NOT_FOUND = "I couldn't find this in the book."

SYSTEM_PROMPT = f"""You are a teaching assistant for the textbook "Introduction to Machine Learning with Python" (Müller & Guido).
Answer the user's QUESTION using ONLY the numbered CONTEXT excerpts taken from that book.

Rules:
1. Use only facts that appear in the CONTEXT. Do NOT use outside knowledge, even if you are sure you know the answer.
2. If the CONTEXT does not contain the answer, reply exactly: "{NOT_FOUND}" You may add one short sentence naming what the excerpts do cover.
3. If the CONTEXT answers only part of the question, answer that part and say clearly what is not covered.
4. Cite the supporting excerpt(s) inline as [1], [2] ... directly after each claim. Never cite an excerpt you did not use.
5. Do not invent numbers, function names, parameters or code. Quote identifiers (e.g. `n_neighbors`, `GridSearchCV`) exactly as they appear in the CONTEXT.
6. Be concise and technically precise: a short paragraph, or a few bullets for lists/steps. No preamble such as "According to the context"."""


def format_context(hits: list[Retrieved]) -> str:
    blocks = []
    for h in hits:
        blocks.append(f"[{h.rank}] Source: {h.chunk.citation()}\n{h.chunk.text}")
    return "\n\n---\n\n".join(blocks)


def build_user_prompt(question: str, hits: list[Retrieved]) -> str:
    return f"CONTEXT:\n{format_context(hits)}\n\nQUESTION: {question}\n\nANSWER (grounded in the CONTEXT only, with [n] citations):"


# ------------------------------------------------------------------------------ providers
# Output budgets are generous because newer models may spend part of it on hidden reasoning.
_MAX_OUT = 2000


def _gemini(model: str, system: str, user: str) -> str:
    from google import genai
    from google.genai import types

    # Gemini 3.x guidance is to leave temperature at its default; earlier families get 0 for determinism.
    cfg = dict(system_instruction=system, max_output_tokens=_MAX_OUT)
    if not model.startswith("gemini-3"):
        cfg["temperature"] = 0.0
    r = genai.Client().models.generate_content(
        model=model, contents=user, config=types.GenerateContentConfig(**cfg))
    return (r.text or "").strip()


def _openai(model: str, system: str, user: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    kw = dict(model=model, max_completion_tokens=_MAX_OUT,
              messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    try:
        r = client.chat.completions.create(temperature=0, **kw)
    except Exception as e:  # some model families only accept the default temperature
        if "temperature" not in str(e).lower():
            raise
        r = client.chat.completions.create(**kw)
    return (r.choices[0].message.content or "").strip()


def _ollama(model: str, system: str, user: str) -> str:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    body = json.dumps({"model": model, "stream": False, "options": {"temperature": 0},
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(f"{host}/api/chat", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())["message"]["content"].strip()


_STOP = set("""a an the of in on to for and or is are was were be been being what which who whom how why when where does do
did can could should would about between with without from by as at it its this that these those vs versus explain
describe difference define mean means used use using tell me""".split())


def _terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z_][a-z0-9_]+", text.lower()) if w not in _STOP and len(w) > 2}


def extractive_answer(question: str, hits: list[Retrieved], max_sentences: int = 4) -> str:
    """Offline baseline: return the most question-relevant *sentences* from the retrieved chunks,
    verbatim, with citations. Cannot hallucinate (it only copies text) but is less fluent than an LLM."""
    q = _terms(question)
    scored = []
    for h in hits:
        prose = [p for p in h.chunk.text.split("\n\n") if not p.lstrip().startswith(("In[", "Out["))]
        for para in prose:
            for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", para):
                s = s.strip()
                # keep only complete prose sentences (drops headings, captions cut by page breaks, "over-")
                if len(s.split()) < 6 or len(s) > 450 or not s.endswith((".", "!", "?")):
                    continue
                if re.match(r"^(Figure|Table)\s+\d", s) or s.count("(") != s.count(")"):
                    continue
                overlap = len(q & _terms(s))
                if overlap:
                    scored.append((overlap / (len(q) ** 0.5), -h.rank, s.strip(), h.rank))
    if not scored:
        return NOT_FOUND
    best, seen = [], set()
    for _, _, s, rank in sorted(scored, reverse=True):
        if s in seen:
            continue
        seen.add(s)
        best.append((s, rank))
        if len(best) == max_sentences:
            break
    # present in book order (by rank) so it reads naturally
    best.sort(key=lambda x: x[1])
    return " ".join(f"{s} [{r}]" for s, r in best)


PROVIDERS = {"gemini": _gemini, "openai": _openai, "ollama": _ollama}


def generate(question: str, hits: list[Retrieved], provider: str, model: str, min_score: float) -> dict:
    """Returns {answer, refused, reason, prompt}. Never calls the LLM when retrieval is weak."""
    if not hits or hits[0].score < min_score:
        return {"answer": NOT_FOUND, "refused": True, "prompt": None,
                "reason": f"best retrieval score {hits[0].score:.3f} < threshold {min_score:.3f}" if hits else "no hits"}
    user = build_user_prompt(question, hits)
    if provider == "extractive":
        ans = extractive_answer(question, hits)
    else:
        ans = PROVIDERS[provider](model, SYSTEM_PROMPT, user)
    return {"answer": ans, "refused": ans.startswith(NOT_FOUND), "reason": "", "prompt": user}


def lexical_support(answer: str, hits: list[Retrieved]) -> float:
    """Cheap grounding signal in [0,1]: share of the answer's content words that occur in the retrieved
    context. 1.0 = every content word of the answer is present in the sources (typical of copied / closely
    paraphrased answers); low values flag possible hallucination. Not a substitute for human review."""
    a = _terms(re.sub(r"\[\d+\]", " ", answer))
    if not a:
        return 1.0
    ctx = _terms(" ".join(h.chunk.text for h in hits))
    return round(len(a & ctx) / len(a), 3)
