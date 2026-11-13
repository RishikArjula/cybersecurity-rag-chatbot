# Cybersecurity Knowledge Assistant – RAG Chatbot

## Project Overview

The Cybersecurity Knowledge Assistant is a Retrieval-Augmented Generation (RAG) chatbot that helps users understand cybersecurity concepts by retrieving relevant information from a cybersecurity knowledge-base PDF and generating answers using a Large Language Model (LLM).

The project uses LangChain, Hugging Face embeddings, FAISS vector database, Groq API, and Streamlit.

## Features

- Loads cybersecurity information from PDF documents.
- Splits documents into smaller text chunks.
- Generates embeddings using Hugging Face.
- Stores and retrieves document embeddings using FAISS.
- Uses a retriever to find relevant information.
- Generates context-based answers using the Groq LLM.
- Provides an interactive chatbot interface using Streamlit.
- Displays source document information for retrieved content.

## Technology Stack

- **Programming Language:** Python
- **Framework:** LangChain
- **PDF Loading:** PyPDFLoader
- **Text Splitting:** RecursiveCharacterTextSplitter
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2
- **Vector Database:** FAISS
- **LLM:** Groq API – openai/gpt-oss-20b
- **Frontend:** Streamlit
- **Environment Variables:** python-dotenv

## Project Structure

```text
cybersecurity_rag/
│
├── data/
│   └── Cybersecurity_Knowledge_Base_RAG.pdf
│
├── vectorstore/       # Generated locally during ingestion
│
├── app.py             # Streamlit chatbot interface
├── chatbot.py         # Command-line chatbot
├── ingest.py          # PDF processing and vector database creation
├── requirements.txt   # Project dependencies
├── .env               # API key (not uploaded)
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
cd cybersecurity_rag
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Replace the placeholder with your own Groq API key.

### 4. Create the Vector Database

Run the ingestion script:

```bash
python ingest.py
```

This process loads the PDF, splits its content, generates embeddings, and saves the FAISS vector database locally.

### 5. Run the Chatbot

Start the Streamlit application:

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to interact with the chatbot.

To run the optional command-line chatbot:

```bash
python chatbot.py
```

## How RAG Works

1. **Document Loading:** Loads cybersecurity information from the PDF.
2. **Text Splitting:** Divides the document into smaller chunks.
3. **Embedding Generation:** Converts text chunks into numerical vectors.
4. **Vector Storage:** Stores embeddings in the FAISS vector database.
5. **Retrieval:** Finds relevant chunks based on the user's question.
6. **Answer Generation:** Sends retrieved context to the LLM to generate an answer.

## Security Notes

- Never upload `.env` files or API keys to GitHub.
- The FAISS vector database is generated locally and can be recreated using `ingest.py`.
- Answers depend on the information available in the knowledge-base document and retrieved context.
- This project is intended for educational and demonstration purposes, not as a replacement for professional cybersecurity advice.

## Author

**Arjula Rishik Reddy**

B.Tech – Computer Science Engineering

## Acknowledgement

Developed as a practical project to understand Retrieval-Augmented Generation (RAG), document processing, vector databases, and Large Language Model integration using LangChain.