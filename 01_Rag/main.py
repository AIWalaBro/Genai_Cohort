# uplaod document and langchain can read the document from the directory
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Qdrant
from langchain_qdrant import Qdrant as QdrantVectorStore
# from langchain_openai import OpenAIEmbeddings
from euriai.langchain_embed import EuriaiEmbeddings

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()
import os
euri = os.getenv("EURI")


# Ensure the path is correct and points to the PDF file and load the documents
filepath = Path(__file__).parent/"nodejs.pdf"
loader = PyPDFLoader(filepath)  
docs = loader.load() 


# split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(docs)


# Embeddings
# embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")

embedding = EuriaiEmbeddings(api_key= euri)  


# Now Ingest the data in vector store we are using Qdrant over here
vector_store = QdrantVectorStore.from_documents(
    documents=split_docs, 
    embedding= embedding,
    collection_name="learning_vectorstore",
    url="http://localhost:6333",  # Ensure Qdrant is running and accessible at this URL
      # Use gRPC for better performance if available
)

