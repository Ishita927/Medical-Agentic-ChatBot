import os, sys
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
INDEX = os.getenv("PINECONE_INDEX", "medguide")

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # 384 dims

def main(pdf_path):
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    if INDEX not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX, dimension=384, metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    docs = PyPDFLoader(pdf_path).load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150
    ).split_documents(docs)
    print(f"{len(docs)} pages → {len(chunks)} chunks")

    PineconeVectorStore.from_documents(chunks, get_embeddings(), index_name=INDEX)
    print("Done.")

if __name__ == "__main__":
    main(sys.argv[1])