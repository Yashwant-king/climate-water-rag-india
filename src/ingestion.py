import os
from tqdm import tqdm
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_and_chunk_docs(data_dir):
    """
    Loads all Markdown and PDF files from the data directory and chunks them.
    """
    documents = []
    
    # 1. Load Markdown files
    md_loader = DirectoryLoader(data_dir, glob="*.md", loader_cls=TextLoader)
    documents.extend(md_loader.load())
    
    # 2. Load PDF files explicitly
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    for pdf in pdf_files:
        loader = PyPDFLoader(os.path.join(data_dir, pdf))
        documents.extend(loader.load())
        
    print(f"Loaded {len(documents)} documents from {data_dir}")
    
    # Recursive splitter for better semantic cohesion
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2", save_path="data/faiss_index"):
    print(f"Generating embeddings using {model_name}...")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(save_path)
    print(f"Successfully saved FAISS index to {save_path}")
    return vector_store

if __name__ == "__main__":
    DATA_DIR = "data"
    # Ensure directory exists
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    chunks = load_and_chunk_docs(DATA_DIR)
    if chunks:
        print(f"Created {len(chunks)} text chunks.")
        create_vector_store(chunks)
    else:
        print("No documents found in the data directory!")
