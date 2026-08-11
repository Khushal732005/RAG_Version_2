"""
rag_engine.py
-------------
All-round ERP AI Assistant

Behavior:
1. General questions -> LLM general knowledge.
2. ERP/document-specific questions -> RAG/document knowledge.
3. If a document-specific question has weak retrieval,
   the assistant does NOT invent the ERP-specific answer.
4. Explicit document requests always prefer document retrieval.
"""

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

from vector import vector_store, RETRIEVE_K


# ============================================================
# CONFIGURATION
# ============================================================

LLM_MODEL = "llama3.2"

# Minimum similarity required for document-grounded answers.
RELEVANCE_THRESHOLD = 0.45

# Router model
router_model = OllamaLLM(model=LLM_MODEL)

# Main answering model
model = OllamaLLM(model=LLM_MODEL)


# ============================================================
# QUESTION ROUTER
# ============================================================

ROUTER_TEMPLATE = """
You are a question classifier for an intelligent ERP assistant.

Classify the user's question into exactly ONE category:

DOCUMENT
GENERAL

Choose DOCUMENT when the user is asking about:
- Our ERP system
- Our company's software
- A specific ERP module
- A process described in company documents
- Fields, screens, forms, workflows or functionality
- Specific business rules
- Specific configuration
- Specific company processes
- Anything that requires information from uploaded documents
- Questions containing phrases such as:
  "according to the document"
  "according to our ERP"
  "in our system"
  "in the ERP"
  "what fields"
  "what is the process"
  "what is the workflow"

Choose GENERAL when the user is asking about:
- General programming
- Python
- Java
- SQL
- AI
- Machine Learning
- LLMs
- RAG
- General ERP concepts
- General business concepts
- Mathematics
- Science
- Technology
- General knowledge
- Explanations that do not require company-specific information

IMPORTANT:
A general ERP question is GENERAL.

Example:
"What is an ERP?" -> GENERAL

A company-specific ERP question is DOCUMENT.

Example:
"What is the employee creation process in our ERP?" -> DOCUMENT

Return ONLY:
DOCUMENT

or

GENERAL

Question:
{question}
"""


router_prompt = ChatPromptTemplate.from_template(ROUTER_TEMPLATE)
router_chain = router_prompt | router_model


# ============================================================
# DOCUMENT ANSWERING PROMPT
# ============================================================

DOC_TEMPLATE = """
You are an expert assistant for the company's ERP system.

Answer the user's question using the supplied ERP documentation.

IMPORTANT RULES:

1. Use the document context as the source of truth for
   company-specific information.

2. Do NOT invent company-specific:
   - module names
   - field names
   - workflows
   - business rules
   - screen names
   - button names
   - configuration
   - processes

3. If the document contains the answer, explain it clearly.

4. If multiple document sections are relevant, combine them.

5. If the question asks for a process or workflow,
   explain it step-by-step.

6. If the document only partially answers the question,
   clearly state what information is available and
   what information is not available.

7. Do not say "I cannot answer" simply because the
   exact wording of the question does not appear in
   the document. Use relevant information from the
   retrieved context.

ERP DOCUMENTATION:
{context}

USER QUESTION:
{question}

Answer:
"""


doc_prompt = ChatPromptTemplate.from_template(DOC_TEMPLATE)
doc_chain = doc_prompt | model


# ============================================================
# GENERAL KNOWLEDGE PROMPT
# ============================================================

GENERAL_TEMPLATE = """
You are an intelligent, helpful, all-round AI assistant.

Answer the user's question using your general knowledge.

You can answer questions about:

- Python
- Java
- SQL
- Programming
- Artificial Intelligence
- Machine Learning
- Deep Learning
- LLMs
- RAG
- APIs
- Databases
- ERP concepts
- Software development
- Mathematics
- Science
- Technology
- Business concepts
- General knowledge

IMPORTANT:

1. Give a useful and complete answer.

2. Do not unnecessarily say:
   "I don't have information in the documents."

3. The absence of information in the company's
   documents does NOT mean that you cannot answer.

4. Do not invent company-specific ERP information.

5. If the user asks for company-specific information
   and you do not have sufficient document evidence,
   clearly say that the company-specific information
   is not available in the provided documentation.

6. Explain concepts simply and provide examples when useful.

USER QUESTION:
{question}

Answer:
"""


general_prompt = ChatPromptTemplate.from_template(GENERAL_TEMPLATE)
general_chain = general_prompt | model


# ============================================================
# ROUTER FUNCTION
# ============================================================

def classify_question(question: str) -> str:
    """
    Classify question as DOCUMENT or GENERAL.
    """

    try:
        result = router_chain.invoke({
            "question": question
        })

        result = str(result).strip().upper()

        if "DOCUMENT" in result:
            return "DOCUMENT"

        return "GENERAL"

    except Exception as e:
        print(f"Router error: {e}")

        # Safe fallback:
        # use retrieval if router fails.
        return "DOCUMENT"


# ============================================================
# DOCUMENT RETRIEVAL
# ============================================================

def retrieve_relevant_chunks(question: str, k: int = RETRIEVE_K):

    results = vector_store.similarity_search_with_relevance_scores(
        question,
        k=k
    )

    if not results:
        return [], False, 0.0

    top_score = max(score for _, score in results)

    chunks = [
        doc
        for doc, score in results
        if score >= RELEVANCE_THRESHOLD
    ]

    is_relevant = len(chunks) > 0

    return (
        chunks,
        is_relevant,
        round(top_score, 3)
    )


# ============================================================
# FORMAT DOCUMENT CONTEXT
# ============================================================

def format_context(chunks):

    return "\n\n".join(
        f"""
[Source: {c.metadata.get('filename', 'unknown')}]

{c.page_content}
"""
        for c in chunks
    )


# ============================================================
# MAIN STREAMING FUNCTION
# ============================================================

def answer_stream(question: str):

    question_type = classify_question(question)

    # --------------------------------------------------------
    # GENERAL QUESTION
    # --------------------------------------------------------

    if question_type == "GENERAL":

        yield {
            "mode": "general",
            "sources": [],
            "score": 0.0
        }

        stream = general_chain.stream({
            "question": question
        })

        for token in stream:
            yield token

        return


    # --------------------------------------------------------
    # DOCUMENT QUESTION
    # --------------------------------------------------------

    chunks, is_relevant, top_score = retrieve_relevant_chunks(
        question
    )

    # --------------------------------------------------------
    # Document found
    # --------------------------------------------------------

    if is_relevant:

        context = format_context(chunks)

        sources = sorted({
            c.metadata.get(
                "filename",
                "unknown"
            )
            for c in chunks
        })

        yield {
            "mode": "document",
            "sources": sources,
            "score": top_score
        }

        stream = doc_chain.stream({
            "context": context,
            "question": question
        })

        for token in stream:
            yield token

        return


    # --------------------------------------------------------
    # Document question but no sufficient evidence
    # --------------------------------------------------------

    yield {
        "mode": "document_not_found",
        "sources": [],
        "score": top_score
    }

    fallback_message = (
        "I couldn't find sufficient information about this "
        "specific ERP functionality in the provided documents. "
        "I don't want to invent company-specific details."
    )

    yield fallback_message


# ============================================================
# NON-STREAMING ANSWER
# ============================================================

def answer(question: str) -> dict:

    gen = answer_stream(question)

    meta = next(gen)

    text = "".join(gen)

    meta["answer"] = text

    return meta

