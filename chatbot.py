
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS

# Step 1: Load Environment Variables
load_dotenv()

# Step 2: Load the Embedding Model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 3: Load the Saved FAISS Vector Database
vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

# Step 4: Create a Retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

# Step 5: Initialize the LLM Model
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

# Step 6: Chatbot Loop
print("Cybersecurity Knowledge Assistant is ready!")
print("Type 'exit' to quit.\n")

while True:
    question = input("You: ").strip()

    if question.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    if not question:
        continue

    # Retrieve Relevant Documents
    documents = retriever.invoke(question)

    # Convert Documents into Context
    context = "\n\n".join(
        [document.page_content for document in documents]
    )

    # Final Prompt to LLM
    final_prompt = f"""
    You are a helpful Cybersecurity Knowledge Assistant.

    Answer the user's question using only the provided context.

    If the answer is not available in the context, say:
    "I don't know based on the provided documents."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    # Generate Answer
    response = llm.invoke(final_prompt)

    print(f"\nAI: {response.content}\n")
