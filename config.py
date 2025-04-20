class Config:
    # Document Configuration
    DOCUMENT_PATHS = [
        "./documents/manual.pdf",  # Add your PDF/document paths here
        # "./documents/guide.docx",
        # "./documents/instructions.txt",
    ]
    
    # Embedding Model Configuration
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Default: good balance of speed/accuracy
    
    # Alternative embedding models (uncomment to use):
    # EMBEDDING_MODEL = "multi-qa-MiniLM-L6-cos-v1"  # QA-optimized
    # EMBEDDING_MODEL = "all-mpnet-base-v2"  # Higher quality but larger
    
    # Chroma DB Configuration
    CHROMA_PERSIST_DIR = "./chroma_db"  # Where to store the vector database
    CHROMA_COLLECTION_NAME = "document_embeddings"  # Collection name in Chroma

    # Processing Configuration
    FILTER_ELEMENTS = ["Footer", "Image", "Header", "PageNumber"]  # Elements to exclude
    UNKNOWN_PAGE_PLACEHOLDER = "unknown"  # How to handle pages without numbers


# Validation
if __name__ == "__main__":
    print("Current Configuration:")
    print(f"Document Paths: {Config.DOCUMENT_PATHS}")
    print(f"Embedding Model: {Config.EMBEDDING_MODEL}")
    print(f"Chroma Persistence Directory: {Config.CHROMA_PERSIST_DIR}")
