# retriever_setup.py
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import Tuple, List
from langchain_core.documents import Document
from config import Config
import uuid

def setup_retrievers(documents: List[Document], elements: List[Document]) -> EnsembleRetriever:
    """
    Creates and configures hybrid (BM25 + Chroma) retriever
    
    Args:
        documents: List of full-page Documents
        elements: List of element-level Documents
        
    Returns:
        Configured EnsembleRetriever instance
    """
    # BM25 Retriever (keyword-based)
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 2  # Return top 2 BM25 matches

    # Chroma Retriever (semantic)
    embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
    chroma_db = Chroma.from_documents(
        documents=elements,
        embedding=embeddings,
        persist_directory=Config.CHROMA_PERSIST_DIR,
        collection_name=Config.CHROMA_COLLECTION_NAME
    )
    chroma_retriever = chroma_db.as_retriever(search_kwargs={"k": 2})

    # Hybrid Ensemble Retriever
    return EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.6, 0.4]  # Tweak based on your needs
    )


if __name__ == "__main__":
    # Example usage (would normally import process_pdf from another module)
    from document_processor import process_pdf  # Assume this exists
    
    # Process documents using config paths
    all_docs = []
    all_elements = []
    for path in Config.DOCUMENT_PATHS:
        docs, elements = process_pdf(path)
        all_docs.extend(docs)
        all_elements.extend(elements)
    
    # Initialize retrievers
    retriever = setup_retrievers(all_docs, all_elements)
    
    # Test query
    test_query = "How to troubleshoot server issues?"
    results = retriever.invoke(test_query)
    print(f"Found {len(results)} relevant documents for: {test_query}")
