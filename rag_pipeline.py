import fitz
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv


def extract_text_from_pdf(pdf_path):
    """Extract all text from a single PDF file"""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text


def extract_text_from_multiple_pdfs(pdf_folder):
    """Stage 1 - Extract text from ALL PDFs in a folder"""
    combined_text = ""
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        print(f"   Reading: {pdf_file}")
        combined_text += extract_text_from_pdf(pdf_path)

    return combined_text


def split_text_into_chunks(text):
    """Stage 2 - Split text into smaller chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_text(text)
    return chunks


def load_embedding_model():
    """Stage 3 - Load sentence transformer embedding model"""
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    return embeddings


def create_vector_store(chunks, embeddings):
    """Stage 4 - Convert chunks to vectors and store in FAISS"""
    print(f"⏳ Embedding {len(chunks)} chunks into FAISS...")
    vector_store = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )
    vector_store.save_local("faiss_index")
    print("✅ Stage 4: Vector store created and saved")
    return vector_store


def load_vector_store(embeddings):
    """Stage 4 - Load existing FAISS index from disk"""
    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_store


def create_qa_chain(vector_store):
    """Stage 5 - Create the RAG question-answering chain"""
    load_dotenv()

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
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
        search_kwargs={"k": 5}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain


if __name__ == "__main__":

    # Stage 1 - Extract from all PDFs in folder
    print("⏳ Extracting text from all PDFs...")
    text = extract_text_from_multiple_pdfs("sample_pdfs")
    print(f"✅ Stage 1: Extracted {len(text)} characters from all PDFs")

    # Stage 2
    chunks = split_text_into_chunks(text)
    print(f"✅ Stage 2: Created {len(chunks)} chunks")

    # Stage 3
    print("\n⏳ Loading embedding model...")
    embeddings = load_embedding_model()
    print(f"✅ Stage 3: Embedding model loaded")

    # Stage 4
    if os.path.exists("faiss_index"):
        print("✅ Stage 4: Loading existing vector store...")
        vector_store = load_vector_store(embeddings)
    else:
        vector_store = create_vector_store(chunks, embeddings)

    # Stage 5
    print("\n⏳ Setting up QA chain...")
    qa_chain = create_qa_chain(vector_store)
    print("✅ Stage 5: QA chain ready\n")

    # Test questions
    questions = [
        "Who is eligible for PM Kisan scheme?",
        "What is the benefit amount under PM Kisan?",
        "Who is excluded from PM Kisan scheme?"
    ]

    for question in questions:
        print(f"Question: {question}")
        result = qa_chain.invoke({"query": question})
        print(f"Answer: {result['result']}")
        print("---")