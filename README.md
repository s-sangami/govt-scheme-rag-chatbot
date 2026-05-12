# 🏛️ Government Scheme Eligibility Chatbot

A RAG-based AI chatbot that allows Indian citizens to upload official government scheme PDFs and ask questions in plain English, Hindi, Tamil, Telugu, Kannada, or Malayalam — and get instant, accurate answers.

---

## 🎯 Problem Statement

Millions of Indians are unaware of government welfare schemes they qualify for. Official documents are long, complex, and written in legal language that most citizens cannot easily understand. This chatbot bridges that gap — making scheme information accessible to everyone in their own language.

---

## 🚀 Features

- 📄 Upload any Indian government scheme PDF
- 🌐 Ask questions in 6 languages — English, Hindi, Tamil, Telugu, Kannada, Malayalam
- 🎯 Accurate answers pulled directly from the uploaded document
- 🚫 Zero hallucination — chatbot only answers from the document, never guesses
- ⚡ Fast semantic search using FAISS vector database
- 🔄 Upload multiple PDFs and switch between them

---

## 🧠 How It Works

```
PDF Upload
    ↓
Text Extraction (PyMuPDF)
    ↓
Text Chunking (LangChain)
    ↓
Embeddings (Sentence Transformers)
    ↓
FAISS Vector Store

User Types Question (any language)
    ↓
Translate to English (Deep Translator)
    ↓
Semantic Search in FAISS
    ↓
Top 5 Relevant Chunks Retrieved
    ↓
Llama 3.1 Generates Answer
    ↓
Translate Answer Back to User Language
    ↓
Answer Displayed
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| PDF Extraction | PyMuPDF (fitz) | Extract raw text from PDF |
| Text Chunking | LangChain RecursiveCharacterTextSplitter | Split text into 500-char chunks |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) | Convert text to 384-dim vectors |
| Vector Database | FAISS | Store and search vectors locally |
| LLM | Llama 3.1 via Groq API | Generate final answers |
| Translation | Deep Translator + LangDetect | Multilingual support |
| UI | Streamlit | Web interface |
| Language | Python 3.11 | Core language |

---

## 📁 Project Structure

```
govt-scheme-rag-chatbot/
│
├── app.py                  ← Streamlit UI (run this)
├── rag_pipeline.py         ← Core RAG logic
├── .env                    ← Your Groq API key (never commit this)
├── .gitignore              ← Ignores venv, .env, faiss_index
├── requirements.txt        ← All dependencies
├── README.md               ← This file
└── sample_pdfs/            ← Place your PDFs here
    └── pm_kisan.pdf
```

---

## ⚙️ Installation & Setup

**Step 1 — Clone the repository**
```bash
git clone https://github.com/YOURUSERNAME/govt-scheme-rag-chatbot
cd govt-scheme-rag-chatbot
```

**Step 2 — Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate
```

**Step 3 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 4 — Set up your Groq API key**

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get your free API key at: https://console.groq.com

**Step 5 — Run the app**
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📄 Example PDFs to Test

- [PM Kisan Samman Nidhi FAQ](https://pmkisan.gov.in/Documents/RevisedFAQ.pdf)
- [PM Kisan Operational Guidelines](https://pmkisan.gov.in/Documents/RevisedPM-KISANOperationalGuidelines(English).pdf)
- Any official PDF from [india.gov.in](https://india.gov.in)

---

## 💬 Example Questions

**English:**
- Who is eligible for PM Kisan scheme?
- What is the benefit amount?
- Who is excluded from this scheme?

**Hindi:**
- पीएम किसान योजना के लिए कौन पात्र है?
- योजना का लाभ क्या है?

**Tamil:**
- யார் தகுதியானவர்கள்?
- பயனாளிகள் யார்?

---

## 🔮 Future Improvements

- Multi-PDF comparison — compare eligibility across multiple schemes simultaneously
- Eligibility form — user fills income, occupation, land details and gets auto-checked result
- Voice input — speak your question instead of typing
- Source citation — show exact page number and section of the answer
- Evaluation — integrate RAGAS framework to measure answer quality

---

## 🧪 Key Concepts Used

- **RAG (Retrieval Augmented Generation)** — combines document retrieval with LLM generation
- **Semantic Search** — finds relevant chunks by meaning, not just keyword matching
- **Vector Embeddings** — converts text to numerical vectors capturing semantic meaning
- **FAISS** — Facebook AI Similarity Search for fast local vector retrieval
- **Prompt Engineering** — constrains LLM to answer only from document context
- **Multilingual NLP** — language detection and neural machine translation

---

## 👨‍💻 Author

**SANGAMI S**
3rd Year B.E. Computer Science Engineering
Mini Project — Deep Learning Subject

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
