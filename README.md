# ERP Documentation Assistant — v2 (Chatbot UI)

A RAG chatbot for your ERP module and process-flow documentation. Answers use
your own `.docx` files first; if a question isn't covered by any document, it
falls back to general knowledge and **clearly labels the answer as such** —
it never silently blends the two or guesses at document content it doesn't have.

## What's new in v2 vs v1

| | v1 | v2 |
|---|---|---|
| Interface | Terminal only | Streamlit chat UI + terminal |
| Unanswerable questions | Model guesses from weak context | Explicit fallback to labeled general knowledge |
| Response delivery | Waits for full answer | Streams token-by-token |
| Source visibility | Printed inline | Tagged chips per message, in chat history |
| Adding documents | Restart required | "Scan for new documents" button, no restart |

## Architecture

```
documents/         <- put your .docx files here (add more anytime)
vector.py           <- indexes documents into a persistent Chroma vector DB
rag_engine.py        <- retrieval + relevance decision + prompts (shared logic)
main.py              <- terminal chat interface
app.py               <- Streamlit chat interface
requirements.txt
```

`vector.py` and `rag_engine.py` contain zero UI code — both `main.py` and
`app.py` call the same `answer_stream()` function, so the two interfaces
always behave identically. If you build a third interface later (Slack bot,
API, etc.), reuse `rag_engine.py` the same way.

### How the document-vs-general decision works

1. Your question is embedded and compared against all indexed chunks using
   cosine similarity, returning a **relevance score from 0 (unrelated) to 1
   (near-identical meaning)**.
2. If the best-matching chunk's score is **≥ `RELEVANCE_THRESHOLD`** (default
   `0.45`, set in `rag_engine.py`), the question is treated as answerable
   from your documents. Only chunks above the threshold are passed as context,
   under a strict "use ONLY this context" prompt.
3. If nothing clears the threshold, the question is treated as out-of-scope
   for your documents. A separate prompt answers from general knowledge,
   explicitly instructed **not** to invent specifics about your company's
   ERP setup (module names, field names, workflows) since it has no evidence
   for those — only genuinely general ERP/domain knowledge.
4. The UI always shows which mode was used, plus the source file(s) or the
   relevance score, so you can judge how much to trust each answer.

This threshold is the main lever for accuracy. Too low → general questions
get incorrectly "answered" from irrelevant document chunks. Too high → valid
document-covered questions get pushed to the general-knowledge fallback.
Adjust `RELEVANCE_THRESHOLD` in `rag_engine.py` and re-test against your own
documents; there's no universal correct value.

## Speed / efficiency notes

- **Singletons, not reloads.** The embedding model client, the Chroma DB
  connection, and the LLM client are all created once at import time
  (`vector.py` / `rag_engine.py`) and reused for every question — including
  across Streamlit re-runs, which happen on every UI interaction. This avoids
  the most common cause of a "slow" local RAG app: re-initializing the model
  or DB connection on every message.
- **Streaming.** Both interfaces use `.stream()` instead of `.invoke()`, so
  tokens appear as they're generated rather than after the full response is
  ready. This doesn't reduce total generation time, but it removes the
  "did it freeze?" dead air on longer answers.
- **Incremental indexing.** `index_new_documents()` only embeds files not
  already in the DB, so adding one new document doesn't re-embed everything
  you already indexed.
- **Chunk size (800 chars) / top-k (5).** Smaller `k` and smaller chunks
  mean less text sent to the LLM per question, which is the single biggest
  lever on local LLM response time. If answers feel slow, try `RETRIEVE_K = 3`
  in `vector.py` first before considering a smaller LLM.

## Setup

### 1. Install Ollama and pull models

https://ollama.com, then:

```bash
ollama pull mxbai-embed-large
ollama pull llama3.2
```

Keep Ollama running in the background.

### 2. Install Python dependencies

```bash
cd erp_rag_v2
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add your documents

Drop `.docx` files into `documents/`:

```
erp_rag_v2/
  documents/
    Inventory_Module.docx
    Procurement_Process_Flow.docx
    Finance_Module_Overview.docx
  vector.py
  rag_engine.py
  main.py
  app.py
```

### 4. Run

**Chat UI (recommended):**
```bash
streamlit run app.py
```
Opens in your browser. Use the sidebar's **Scan for new documents** button
after adding more `.docx` files — no restart needed.

**Terminal version:**
```bash
python main.py
```

## Adding documents later

Copy new `.docx` files into `documents/`, then either click **Scan for new
documents** in the sidebar (UI) or just re-run `python main.py` (CLI). Only
the new files get embedded; existing ones aren't touched.

## Editing an existing document

Indexing is tracked by file path, not content, so editing an already-indexed
file won't auto-reindex. Force it manually:

```python
from vector import reindex_file
reindex_file("./documents/Inventory_Module.docx")
```

Or delete `chrome_langchain_db/` to reset everything and re-index from scratch.

## Known limitations

- The relevance threshold is a heuristic, not a guarantee — it can still
  occasionally misclassify a borderline question. Watch the displayed score
  and adjust the threshold if you see a pattern of misfires.
- `llama3.2` (via Ollama) is a small local model; for higher-quality answers,
  swap `LLM_MODEL` in `rag_engine.py` for a larger local model or a hosted
  API model, at the cost of speed.
- General-knowledge answers are still LLM output and can be wrong — the
  "general knowledge" label means "not grounded in your documents," not
  "guaranteed correct."
