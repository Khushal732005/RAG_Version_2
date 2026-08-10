"""
rag_engine.py
-------------
The actual RAG decision logic, kept separate from any UI. Both main.py (CLI)
and app.py (Streamlit chat UI) import this, so the behavior is identical in
both places and you only tune prompts/thresholds in one file.

Core idea (anti-hallucination):
  1. Retrieve the top-k chunks and their relevance scores (0 = unrelated,
     1 = near-perfect match, thanks to cosine distance configured in vector.py).
  2. If the best score clears RELEVANCE_THRESHOLD, treat the question as
     answerable from your documents -> use a strict "context-only" prompt.
  3. If it doesn't, treat it as out-of-scope for your documents -> use a
     separate "general knowledge" prompt that explicitly forbids inventing
     ERP-specific specifics it has no evidence for.
  4. The UI always tells the user which mode was used, so answers are never
     silently mixed or misattributed.
"""

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

from vector import vector_store, RETRIEVE_K

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LLM_MODEL = "llama3.2"

# Tune this against your own documents. Chroma's normalized relevance score
# is roughly: 1.0 = near-identical meaning, 0.5 = loosely related, 0.0 = unrelated.
# Start at 0.45-0.5 and adjust based on false-positives/negatives you observe.
RELEVANCE_THRESHOLD = 0.45

# Loaded once at import time (this is the biggest speed lever - the model
# client is reused across every question instead of being re-created per call).
model = OllamaLLM(model=LLM_MODEL)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
DOC_TEMPLATE = """You are an expert assistant for our company's ERP system.
Answer the question using ONLY the context excerpts below. Do not add facts
that are not supported by the context. If the context only partially answers
the question, explicitly say what is missing rather than filling the gap.

Context from ERP documentation:
{context}

Question: {question}

If this is about a process flow, answer step-by-step in order. Mention which
document(s) the answer is based on.
"""

GENERAL_TEMPLATE = """You are a helpful, accurate general-knowledge assistant.
The user's question could not be matched to anything in the internal ERP
documentation, so answer using your own general knowledge instead.

Rules:
- Be factually careful. If you're not confident about something, say so
  rather than guessing.
- Do NOT invent specifics about THIS company's ERP setup - module names,
  field names, button labels, or workflows - since you have no document
  evidence for those. Stick to general, widely-applicable knowledge
  (e.g. how ERP systems typically handle this).
- Be concise and directly useful.

Question: {question}
"""

doc_chain = ChatPromptTemplate.from_template(DOC_TEMPLATE) | model
general_chain = ChatPromptTemplate.from_template(GENERAL_TEMPLATE) | model


def retrieve_relevant_chunks(question: str, k: int = RETRIEVE_K):
    """
    Returns (chunks, is_relevant, top_score).
    chunks: the Document objects to use as context (empty if none pass the bar)
    is_relevant: whether the top match clears RELEVANCE_THRESHOLD
    top_score: the best relevance score seen, for display/debugging
    """
    results = vector_store.similarity_search_with_relevance_scores(question, k=k)
    if not results:
        return [], False, 0.0

    top_score = max(score for _, score in results)
    is_relevant = top_score >= RELEVANCE_THRESHOLD
    chunks = [doc for doc, score in results if score >= RELEVANCE_THRESHOLD]
    return chunks, is_relevant, round(top_score, 3)


def format_context(chunks) -> str:
    return "\n\n".join(
        f"[Source: {c.metadata.get('filename', 'unknown')}]\n{c.page_content}"
        for c in chunks
    )


def answer_stream(question: str):
    """
    Generator. First yield is always a metadata dict:
        {"mode": "document"|"general", "sources": [...], "score": float}
    Every subsequent yield is a text token to append to the answer.
    Streaming (rather than waiting for the full response) is what makes the
    UI feel fast even though local LLMs generate token-by-token.
    """
    chunks, is_relevant, top_score = retrieve_relevant_chunks(question)

    if is_relevant:
        context = format_context(chunks)
        sources = sorted({c.metadata.get("filename", "unknown") for c in chunks})
        yield {"mode": "document", "sources": sources, "score": top_score}
        stream = doc_chain.stream({"context": context, "question": question})
    else:
        yield {"mode": "general", "sources": [], "score": top_score}
        stream = general_chain.stream({"question": question})

    for token in stream:
        yield token


def answer(question: str) -> dict:
    """Non-streaming convenience wrapper (used by the CLI). Returns full answer + metadata."""
    gen = answer_stream(question)
    meta = next(gen)
    text = "".join(gen)
    meta["answer"] = text
    return meta
