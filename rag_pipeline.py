import fitz
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv


def extract_pages_from_pdf(pdf_path):
    """Extract text page by page. Returns a list of (page_number, text), 1-based."""
    doc = fitz.open(pdf_path)
    pages = [(i + 1, page.get_text()) for i, page in enumerate(doc)]
    doc.close()
    return pages


def split_pages_into_documents(pages, source=None):
    """Stage 2 - Split each page into chunks, keeping the page number as metadata"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )
    documents = []
    for page_number, text in pages:
        metadata = {"page": page_number}
        if source:
            metadata["source"] = source
        documents.extend(splitter.create_documents([text], metadatas=[metadata]))
    return documents


def load_embedding_model():
    """Stage 3 - Load sentence transformer embedding model"""
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    return embeddings


def create_vector_store(documents, embeddings):
    """Stage 4 - Convert chunk documents to vectors and store in FAISS"""
    print(f"⏳ Embedding {len(documents)} chunks into FAISS...")
    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )
    print("✅ Stage 4: Vector store created (in memory)")
    return vector_store


REFUSAL_TEXT = "I could not find this information in the document."


def is_refusal(answer):
    """True if the answer is the standard 'not in the document' refusal."""
    return REFUSAL_TEXT.lower().rstrip(".") in answer.lower()


def create_qa_chain(vector_store):
    """Stage 5 - Create the RAG question-answering chain"""
    load_dotenv()

    llm = ChatGroq(
        model_name=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt_template = """
    You are a helpful assistant for Indian government schemes.
    Use ONLY the context below to answer the question.
    If the answer is not in the context, say
    "I could not find this information in the document."

    Context:
    {context}

    Question: {question}

    Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 8}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain


if __name__ == "__main__":

    # Stage 1 - Extract pages from every PDF in the folder
    print("⏳ Extracting text from all PDFs...")
    documents = []
    for name in sorted(os.listdir("sample_pdfs")):
        if name.lower().endswith(".pdf"):
            pages = extract_pages_from_pdf(os.path.join("sample_pdfs", name))
            documents.extend(split_pages_into_documents(pages, source=name))
    print(f"✅ Stages 1-2: Created {len(documents)} chunks")

    # Stage 3
    print("\n⏳ Loading embedding model...")
    embeddings = load_embedding_model()
    print("✅ Stage 3: Embedding model loaded")

    # Stage 4 (in memory only)
    vector_store = create_vector_store(documents, embeddings)

    # Stage 5
    print("\n⏳ Setting up QA chain...")
    qa_chain = create_qa_chain(vector_store)
    print("✅ Stage 5: QA chain ready\n")

    questions = [
        "Who is eligible for PM Kisan scheme?",
        "What is the benefit amount under PM Kisan?",
        "Who is excluded from PM Kisan scheme?"
    ]

    for question in questions:
        print(f"Question: {question}")
        result = qa_chain.invoke({"query": question})
        print(f"Answer: {result['result']}")
        pages = sorted({d.metadata["page"] for d in result["source_documents"]})
        print(f"Sources: page {', '.join(map(str, pages))}")
        print("---")
