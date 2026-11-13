
import os
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS

# --------------------------------
# 1. Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Cybersecurity Knowledge Assistant",
    page_icon="🛡️",
    layout="wide"
)

load_dotenv()

# --------------------------------
# 2. Custom Styling
# --------------------------------

st.markdown("""
<style>
    .main {
        background-color: #f5f7fb;
    }

    .block-container {
        padding-top: 2rem;
        max-width: 1000px;
    }

    .main-title {
        font-size: 32px;
        font-weight: 700;
        color: #15365c;
    }

    .subtitle {
        color: #64748b;
        font-size: 15px;
    }

    .stChatMessage {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------
# 3. Header
# --------------------------------

st.markdown(
    '<div class="main-title">🛡️ Cybersecurity Knowledge Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Ask questions about cybersecurity threats, vulnerabilities, SOC operations and incident response.</div>',
    unsafe_allow_html=True
)

st.divider()

# --------------------------------
# 4. Load RAG Components
# --------------------------------

@st.cache_resource
def load_rag_components():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )

    return retriever, llm


# --------------------------------
# 5. Initialize Chat History
# --------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------------
# 6. Sidebar
# --------------------------------

with st.sidebar:

    st.title("🛡️ Cybersecurity Assistant")

    st.write("Your AI-powered cybersecurity learning companion.")

    st.divider()

    st.subheader("Suggested Questions")

    suggestions = [
        "What is phishing?",
        "Explain ransomware attacks.",
        "What is the CIA triad?",
        "Explain the incident response lifecycle.",
        "What is MITRE ATT&CK?"
    ]

    for suggestion in suggestions:
        if st.button(suggestion, use_container_width=True):
            st.session_state.pending_question = suggestion

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Knowledge source: Cybersecurity Knowledge Base PDF")

# --------------------------------
# 7. Display Previous Messages
# --------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("View source documents"):
                for source in message["sources"]:
                    st.write(source)

# --------------------------------
# 8. Chat Input
# --------------------------------

question = st.chat_input("Ask a cybersecurity question...")

if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")

# --------------------------------
# 9. Generate RAG Answer
# --------------------------------

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        try:

            if not os.getenv("GROQ_API_KEY"):
                st.error("GROQ_API_KEY is missing. Please configure your .env file.")
                st.stop()

            if not os.path.exists("vectorstore/index.faiss"):
                st.error("FAISS database not found. Run ingest.py first.")
                st.stop()

            with st.spinner("Searching cybersecurity knowledge..."):

                retriever, llm = load_rag_components()

                # Retrieve relevant documents
                documents = retriever.invoke(question)

                # Build context
                context = "\n\n".join(
                    [doc.page_content for doc in documents]
                )

                if not context.strip():
                    answer = "I don't know based on the provided documents."
                    sources = []

                else:

                    final_prompt = f"""
                    You are a Cybersecurity Knowledge Assistant.

                    Answer the user's question using only the
                    information in the provided context.

                    Do not invent facts or unsupported information.

                    If the answer is unavailable in the context, say:
                    "I don't know based on the provided documents."

                    Context:
                    {context}

                    Question:
                    {question}

                    Answer:
                    """

                    response = llm.invoke(final_prompt)
                    answer = response.content

                    sources = []

                    for doc in documents:
                        page = doc.metadata.get("page", "Unknown")
                        source = doc.metadata.get("source", "Unknown")
                        sources.append(
                            f"{os.path.basename(source)} — Page {page + 1 if isinstance(page, int) else page}"
                        )

                    sources = list(dict.fromkeys(sources))

            st.markdown(answer)

            if sources:
                with st.expander("📚 View source documents"):
                    for source in sources:
                        st.write(f"- {source}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
