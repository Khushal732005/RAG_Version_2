"""
main.py (v2)
------------
Terminal chat loop, now using rag_engine so behavior matches the Streamlit UI exactly.
"""

from rag_engine import answer_stream

print("ERP Assistant (CLI) - type 'q' to quit\n")

while True:
    print("\n-------------------------------")
    question = input("Ask about the ERP system (q to quit): ").strip()
    if question.lower() == "q":
        break
    if not question:
        continue

    stream = answer_stream(question)
    meta = next(stream)

    print()
    for token in stream:
        print(token, end="", flush=True)
    print("\n")

    if meta["mode"] == "document":
        print(f"[Source: {', '.join(meta['sources'])} | relevance {meta['score']}]")
    else:
        print(f"[General knowledge - not found in your documents | relevance {meta['score']}]")
