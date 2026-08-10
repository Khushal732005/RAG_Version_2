"""
vector.py (v2)
---------------
Same indexing job as v1: scan ./documents for .docx files, chunk them,
embed only the ones not already indexed, store in a persistent Chroma DB.

New in v2: the collection is explicitly configured to use cosine distance,
so we can ask Chroma for a normalized 0-1 relevance score per result. That
score is what rag_engine.py uses to decide "answer from documents" vs.
"fall back to general knowledge" - so getting this right matters.
"""

import os
import glob

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DOCUMENTS_FOLDER = "./documents"
DB_LOCATION = "./chrome_langchain_db"
COLLECTION_NAME = "erp_module_docs"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
RETRIEVE_K = 5

# ---------------------------------------------------------------------------
# Embeddings + vector store - created once at import time and reused
# everywhere (this is the main speed win: no re-loading per question).
# ---------------------------------------------------------------------------
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=DB_LOCATION,
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"},  # needed for normalized relevance scores
)


def get_indexed_sources() -> set:
    existing = vector_store.get(include=["metadatas"])
    return {
        meta["source"]
        for meta in existing.get("metadatas", [])
        if meta and "source" in meta
    }


def load_and_split_docx(filepath: str):
    loader = Docx2txtLoader(filepath)
    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(raw_docs)

    for chunk in chunks:
        chunk.metadata["source"] = filepath
        chunk.metadata["filename"] = os.path.basename(filepath)

    return chunks


def index_new_documents():
    """Scan documents/, embed only files not already in the DB. Safe to call anytime."""
    os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)

    all_docx_files = glob.glob(os.path.join(DOCUMENTS_FOLDER, "*.docx"))
    indexed_sources = get_indexed_sources()
    new_files = [f for f in all_docx_files if f not in indexed_sources]

    if not all_docx_files:
        print(f"No .docx files found in {DOCUMENTS_FOLDER}/ yet.")
        return 0

    if not new_files:
        print(f"All {len(all_docx_files)} document(s) already indexed.")
        return 0

    print(f"Indexing {len(new_files)} new document(s)...")
    for filepath in new_files:
        chunks = load_and_split_docx(filepath)
        if not chunks:
            continue
        ids = [f"{os.path.basename(filepath)}::{i}" for i in range(len(chunks))]
        vector_store.add_documents(documents=chunks, ids=ids)
        print(f"  Indexed: {os.path.basename(filepath)} ({len(chunks)} chunks)")

    return len(new_files)


def reindex_file(filepath: str):
    """Force re-embed a single file whose content changed."""
    existing = vector_store.get(include=["metadatas"])
    ids_to_delete = [
        id_ for id_, meta in zip(existing["ids"], existing["metadatas"])
        if meta and meta.get("source") == filepath
    ]
    if ids_to_delete:
        vector_store.delete(ids=ids_to_delete)

    chunks = load_and_split_docx(filepath)
    ids = [f"{os.path.basename(filepath)}::{i}" for i in range(len(chunks))]
    vector_store.add_documents(documents=chunks, ids=ids)
    print(f"Re-indexed {filepath} ({len(chunks)} chunks)")


# Index whatever's new every time this module loads.
index_new_documents()
