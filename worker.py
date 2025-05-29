import os
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_llm():
    # Use a free HuggingFaceHub model (requires HuggingFace API token)
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ.get("HF_TOKEN", "")
    return HuggingFaceHub(
        repo_id="google/flan-t5-small",
        model_kwargs={"temperature": 0.0, "max_length": 256}
    )

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": DEVICE}
    )

def process_document(document_path, vectordb_dir):
    """Loads/splits/embeds the PDF and stores in a persistent Chroma DB."""
    loader = PyPDFLoader(document_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(documents)
    embeddings = get_embeddings()
    # Persist to a directory named after the PDF file (no extension)
    db = Chroma.from_documents(texts, embedding=embeddings, persist_directory=vectordb_dir)
    db.persist()

def get_qa_chain(vectordb_dir):
    """Loads Chroma DB from disk and returns a RetrievalQA chain."""
    embeddings = get_embeddings()
    db = Chroma(persist_directory=vectordb_dir, embedding_function=embeddings)
    llm = get_llm()
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.25}),
        return_source_documents=False,
        input_key="question"
    )
    return chain