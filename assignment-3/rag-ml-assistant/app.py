"""Part 5 - Streamlit Q&A app.   Run:  streamlit run app.py"""
from pathlib import Path

import streamlit as st

from rag.config import ROOT, Config, default_llm_model
from rag.pipeline import RagPipeline, build_index
from rag.retriever import RagIndex

st.set_page_config(page_title="ML Knowledge Assistant", page_icon="📘", layout="wide")
st.title("📘 ML Knowledge Assistant")
st.caption("Retrieval-Augmented Generation over *Introduction to Machine Learning with Python* "
           "(Müller & Guido). Answers are grounded in retrieved book passages and cite chapter/page.")

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    index_dir = st.text_input("Index folder", "index")
    provider = st.selectbox("Answer generator", ["gemini", "openai", "ollama", "extractive"],
                            help="'extractive' needs no API key (copies the best sentences; weaker).")
    llm_model = st.text_input("LLM model", default_llm_model(provider), key=f"model_{provider}")
    k = st.slider("Top-k chunks", 1, 10, 5)
    show_chunks = st.checkbox("Show retrieved passages", True)
    st.divider()
    st.subheader("Index")
    backend = st.selectbox("Embedding backend (for building)", ["st", "openai", "gemini", "tfidf"])
    if st.button("(Re)build index from PDF"):
        with st.spinner("Cleaning, chunking, embedding..."):
            build_index(Config(index_dir=index_dir, embedding_backend=backend), verbose=False)
        st.cache_resource.clear()
        st.success("Index built.")


@st.cache_resource(show_spinner="Loading index...")
def load_index(path: str) -> RagIndex:
    return RagIndex.load(ROOT / path if not Path(path).is_absolute() else Path(path))


try:
    index = load_index(index_dir)
except FileNotFoundError:
    st.warning("No index found. Put the book PDF at `data/intro-to-ml-book.pdf`, pick an embedding backend "
               "in the sidebar and click **(Re)build index from PDF**.")
    st.stop()

m = index.manifest
with st.sidebar:
    st.caption(f"{m['n_chunks']} chunks · {m['embedding_backend']}/{m['embedding_model']} · {m['store']}")
    min_score = st.slider("Refusal threshold (min similarity)", 0.0, 1.0, float(m.get("min_score", 0.0)), 0.01,
                          help="If the best chunk scores below this, the app says it can't find the answer.")

pipe = RagPipeline(index, provider, llm_model, min_score, k)

# ----------------------------------------------------------------------------- chat
if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant"):
        st.write(turn["a"])
        if turn["refused"]:
            st.info("No sufficiently relevant passage was found, so the assistant declined to guess.")
        with st.expander(f"Sources ({len(turn['sources'])})"):
            for s in turn["sources"]:
                st.markdown(f"**[{s['rank']}]** {s['cite']} — similarity `{s['score']:.3f}`")
                if show_chunks:
                    st.text(s["text"][:1200])

examples = ["What is overfitting?", "How does a random forest reduce overfitting?",
            "What is the difference between precision and recall?", "How does k-means clustering work?"]
cols = st.columns(len(examples))
clicked = next((e for c, e in zip(cols, examples) if c.button(e, use_container_width=True)), None)
question = st.chat_input("Ask a machine-learning question...") or clicked

if question:
    with st.spinner("Retrieving and generating..."):
        try:
            r = pipe.ask(question, k)
        except Exception as e:  # missing API key, network error, ...
            st.error(f"Generation failed ({type(e).__name__}): {e}")
            st.stop()
    st.session_state.history.append({
        "q": question, "a": r.answer, "refused": r.refused,
        "sources": [{"rank": h.rank, "score": h.score, "cite": h.chunk.citation(), "text": h.chunk.text} for h in r.hits]})
    st.rerun()
