
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# step1 : load the PDF
loader=PyPDFLoader("data/Cybersecurity_Knowledge_Base_RAG.pdf")
documents=loader.load()

# step 2: split it into Chunks
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks=text_splitter.split_documents(documents)

print(len(chunks))

# step 3: Create Embedding model
embedding_model=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
                                      )

# Step 4: Create vector FAISS database
vectorstore=FAISS.from_documents(chunks,embedding_model)

# Step 5 : Save the vector Database
vectorstore.save_local('vectorstore')

print("Vector Database Created Successfully")