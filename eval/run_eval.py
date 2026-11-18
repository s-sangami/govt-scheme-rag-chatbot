"""Run eval/questions.json through the RAG chain and print a pass/fail table.

A question passes when every expected fact is found in the answer. Each fact is
a list of accepted spellings; one match per fact is enough. Matching is
case-insensitive and collapses whitespace.

The "Retr" column says whether every expected fact appears in the retrieved
chunks (YES/NO), which separates retrieval failures from generation failures.
Set EVAL_SHOW_CHUNKS=1 to print the retrieved chunks of failing questions.

Usage (from the project root):  python eval/run_eval.py
Requires GROQ_API_KEY in the environment or in .env.
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, ".env"))

from rag_pipeline import (  # noqa: E402
    extract_pages_from_pdf,
    split_pages_into_documents,
    load_embedding_model,
    create_vector_store,
    create_qa_chain,
)

PDF_PATH = os.path.join(ROOT, "sample_pdfs", "pm_kisan.pdf")
QUESTIONS_PATH = os.path.join(ROOT, "eval", "questions.json")


def normalise(text):
    return re.sub(r"\s+", " ", text.lower())


def check_facts(answer, expected_facts):
    """Return the list of facts (as 'a | b') that were NOT found in the answer."""
    answer = normalise(answer)
    missing = []
    for alternatives in expected_facts:
        if not any(normalise(alt) in answer for alt in alternatives):
            missing.append(" | ".join(alternatives))
    return missing


def main():
    if not os.getenv("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY is not set (put it in .env or the environment).")

    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    print("Building index from", os.path.relpath(PDF_PATH, ROOT))
    pages = extract_pages_from_pdf(PDF_PATH)
    documents = split_pages_into_documents(pages, source=os.path.basename(PDF_PATH))
    chain = create_qa_chain(create_vector_store(documents, load_embedding_model()))

    rows = []
    for q in questions:
        try:
            result = chain.invoke({"query": q["question"]})
            answer = result["result"]
            pages_used = sorted({d.metadata["page"] for d in result["source_documents"]})
            missing = check_facts(answer, q["expected_facts"])
            context = " ".join(d.page_content for d in result["source_documents"])
            retrieved = "YES" if not check_facts(context, q["expected_facts"]) else "NO"
            chunks = [(d.metadata["page"], d.page_content) for d in result["source_documents"]]
            status = "PASS" if not missing else "FAIL"
            note = "" if not missing else "missing: " + "; ".join(missing)
        except Exception as e:  # keep going so one API error doesn't hide the rest
            answer, pages_used, status, note = "", [], "ERROR", f"{type(e).__name__}: {e}"
            retrieved, chunks = "?", []
        rows.append((q["id"], status, pages_used, q["question"], note, answer, retrieved, chunks))

    print()
    print(f"{'#':>2}  {'Result':<6}  {'Retr':<4}  {'Pages':<14}  Question / notes")
    print("-" * 100)
    for qid, status, pages_used, question, note, _, retrieved, _ in rows:
        pages_text = ",".join(map(str, pages_used)) or "-"
        print(f"{qid:>2}  {status:<6}  {retrieved:<4}  {pages_text:<14}  {question[:70]}")
        if note:
            print(f"{'':>32}{note[:120]}")
    passed = sum(r[1] == "PASS" for r in rows)
    print("-" * 100)
    print(f"{passed}/{len(rows)} passed")

    if passed != len(rows):
        print("\nAnswers for non-passing questions:")
        for qid, status, _, question, _, answer, retrieved, chunks in rows:
            if status != "PASS":
                print(f"[{qid}] (retrieved={retrieved}) {answer.strip()[:400]}")
                if os.getenv("EVAL_SHOW_CHUNKS"):
                    for page, text in chunks:
                        print(f"    --- page {page}: {' '.join(text.split())}")
    sys.exit(0 if passed == len(rows) else 1)


if __name__ == "__main__":
    main()
