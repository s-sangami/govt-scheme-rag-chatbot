import streamlit as st
import os
import logging
from translation import translate_to_english, translate_from_english
from rag_pipeline import (
    extract_pages_from_pdf,
    split_pages_into_documents,
    load_embedding_model,
    create_vector_store,
    create_qa_chain,
    is_refusal
)

logger = logging.getLogger(__name__)

EMPTY_ANSWER_TEXT = "The model returned an empty answer. Please try asking again."
TRANSLATION_DOWN_TEXT = (
    "The translation service is unavailable right now, so your question could not "
    "be translated. Please try again in a minute, or ask in English."
)


# ── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title="Government Scheme Chatbot",
    page_icon="🏛️",
    layout="centered"
)

# ── Header ───────────────────────────────────────────────
st.title("🏛️ Government Scheme Eligibility Chatbot")
st.markdown("Upload any Indian government scheme PDF and ask questions in **English, Hindi, Tamil, Telugu, Kannada or Malayalam.**")
st.divider()

# ── Load embedding model once ────────────────────────────
@st.cache_resource
def get_embedding_model():
    return load_embedding_model()

embeddings = get_embedding_model()

# ── Session state setup ──────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if "char_count" not in st.session_state:
    st.session_state.char_count = 0

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Upload a Government Scheme PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:
        if not st.session_state.pdf_processed:
            with st.spinner("Processing PDF..."):

                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                pages = extract_pages_from_pdf(temp_path)
                chunks = split_pages_into_documents(pages, source=uploaded_file.name)
                vector_store = create_vector_store(chunks, embeddings)
                st.session_state.qa_chain = create_qa_chain(vector_store)
                st.session_state.pdf_processed = True
                st.session_state.chunk_count = len(chunks)
                st.session_state.char_count = sum(len(t) for _, t in pages)

                os.remove(temp_path)

            st.success(f"✅ Ready! {st.session_state.chunk_count} chunks indexed.")
            st.info(f"📊 Characters extracted: {st.session_state.char_count}")

    if st.session_state.pdf_processed:
        if st.button("🔄 Upload New PDF"):
            st.session_state.pdf_processed = False
            st.session_state.chat_history = []
            st.session_state.qa_chain = None
            st.session_state.chunk_count = 0
            st.session_state.char_count = 0
            st.rerun()

    st.divider()
    st.markdown("**How to use:**")
    st.markdown("1. Upload a scheme PDF")
    st.markdown("2. Wait for processing")
    st.markdown("3. Ask your question")
    st.markdown("4. Get instant answers")

    st.divider()
    st.markdown("**Supported Languages:**")
    st.markdown("🇬🇧 English")
    st.markdown("🇮🇳 Hindi | Tamil | Telugu")
    st.markdown("🇮🇳 Kannada | Malayalam")

    st.divider()
    st.markdown("**Example Questions:**")
    st.markdown("- Who is eligible for this scheme?")
    st.markdown("- पात्रता मानदंड क्या हैं? *(Hindi)*")
    st.markdown("- யார் தகுதியானவர்கள்? *(Tamil)*")


# ── Main Chat Area ───────────────────────────────────────
if not st.session_state.pdf_processed:
    st.info("👈 Please upload a Government Scheme PDF from the sidebar to begin.")
    st.markdown("### Example PDFs to try:")
    st.markdown("- 📄 PM Kisan Samman Nidhi Guidelines")
    st.markdown("- 📄 Ayushman Bharat Scheme Document")
    st.markdown("- 📄 PM Mudra Loan Scheme Guidelines")
    st.markdown("- 📄 Any official India.gov.in PDF")

else:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                st.caption(message["sources"])

    user_question = st.chat_input("Ask a question in any language...")

    if user_question:

        with st.chat_message("user"):
            st.markdown(user_question)

        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })

        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):

                english_question, detected_lang, in_ok = translate_to_english(
                    user_question
                )

                lang_names = {
                    "en": "English",
                    "hi": "Hindi",
                    "ta": "Tamil",
                    "te": "Telugu",
                    "kn": "Kannada",
                    "ml": "Malayalam"
                }

                if not in_ok and detected_lang != "en":
                    msg = TRANSLATION_DOWN_TEXT
                    st.error(msg)
                    st.session_state.chat_history.append({
                        "role": "assistant", "content": msg, "sources": ""
                    })
                    st.stop()

                if detected_lang != "en":
                    st.caption(
                        f"🌐 Detected: {lang_names.get(detected_lang, detected_lang)} → Searching in English"
                    )

                result = st.session_state.qa_chain.invoke(
                    {"query": english_question}
                )
                english_answer = (result["result"] or "").strip()
                if not english_answer:
                    english_answer = EMPTY_ANSWER_TEXT

                final_answer, out_ok = translate_from_english(
                    english_answer, detected_lang
                )

            if not in_ok:
                st.caption("⚠️ Could not translate your question; it was searched as written.")
            if not out_ok:
                st.caption("⚠️ Could not translate the answer back; showing it in English.")

            st.markdown(final_answer)

            pages_used = [] if (english_answer == EMPTY_ANSWER_TEXT or is_refusal(english_answer)) else sorted({
                doc.metadata["page"] for doc in result["source_documents"]
            })
            sources_text = (
                "Sources: page " + ", ".join(str(p) for p in pages_used)
                if pages_used else ""
            )
            if sources_text:
                st.caption(sources_text)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": final_answer,
            "sources": sources_text
        })