"""
app.py
------
Streamlit chat UI for the ERP RAG assistant.

Run with:
    streamlit run app.py

Design choices for speed + trust:
- vector.py's embeddings/model/vector_store are module-level singletons, so
  Streamlit's re-run-on-every-interaction behavior doesn't reload anything
  slow (models, DB connections) - it just reuses what's already in memory.
- Responses stream token-by-token (st.empty() + manual updates) instead of
  waiting for the full answer, so the UI feels fast even on a local LLM.
- Every answer is visibly labeled "from your documents" (with source files
  and a relevance score) or "general knowledge" (fallback), so you always
  know how much to trust it - never a silent guess.
"""

import streamlit as st

from rag_engine import answer_stream, RELEVANCE_THRESHOLD
from vector import index_new_documents, get_indexed_sources

st.set_page_config(page_title="ERP Assistant", page_icon="🧭", layout="centered")

# --- light custom styling (kept minimal - this is an internal tool, not a marketing page) ---
st.markdown("""
<style>
.source-tag {
    display: inline-block;
    background: #eef2f7;
    color: #33475b;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.78rem;
    margin-top: 4px;
    margin-right: 4px;
}
.general-tag {
    display: inline-block;
    background: #fff4e5;
    color: #8a5300;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.78rem;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

st.title("🧭 ERP Documentation Assistant")
st.caption(
    "Answers your ERP module and process-flow questions from your documents. "
    "If nothing relevant is found, it clearly labels the answer as general knowledge instead of guessing."
)

# --- sidebar: index management ---
with st.sidebar:
    st.header("📁 Document Index")
    indexed = get_indexed_sources()
    st.write(f"**{len(indexed)}** file(s) indexed" if indexed else "No documents indexed yet.")
    if indexed:
        with st.expander("Show indexed files"):
            for src in sorted(indexed):
                st.write(f"- {src.split('/')[-1]}")

    if st.button("🔄 Scan for new documents", use_container_width=True):
        with st.spinner("Scanning documents/ folder..."):
            added = index_new_documents()
        if added:
            st.success(f"Indexed {added} new file(s).")
        else:
            st.info("Nothing new to index.")
        st.rerun()

    st.divider()
    st.caption(f"Relevance threshold: {RELEVANCE_THRESHOLD} (tune in rag_engine.py)")
    st.caption("Drop new `.docx` files into the `documents/` folder, then click **Scan** — no restart needed.")

# --- chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("mode") == "document":
            tags = "".join(f'<span class="source-tag">📄 {s}</span>' for s in msg["sources"])
            st.markdown(f"{tags} <span class='source-tag'>relevance {msg['score']}</span>", unsafe_allow_html=True)
        elif msg.get("mode") == "general":
            st.markdown(
                f"<span class='general-tag'>🌐 General knowledge — not found in your documents "
                f"(relevance {msg['score']})</span>",
                unsafe_allow_html=True,
            )

# --- chat input ---
question = st.chat_input("Ask about the ERP system...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        with st.spinner("Searching your documents..."):
            stream = answer_stream(question)
            meta = next(stream)  # metadata always comes first

        for token in stream:
            full_response += token
            placeholder.markdown(full_response + "▌")
        placeholder.markdown(full_response)

        if meta["mode"] == "document":
            tags = "".join(f'<span class="source-tag">📄 {s}</span>' for s in meta["sources"])
            st.markdown(f"{tags} <span class='source-tag'>relevance {meta['score']}</span>", unsafe_allow_html=True)
        else:
            st.markdown(
                f"<span class='general-tag'>🌐 General knowledge — not found in your documents "
                f"(relevance {meta['score']})</span>",
                unsafe_allow_html=True,
            )

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "mode": meta["mode"],
        "sources": meta.get("sources", []),
        "score": meta["score"],
    })
