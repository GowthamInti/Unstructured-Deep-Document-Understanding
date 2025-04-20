# Unstructured-Deep-Document-Understanding
A hybrid retrieval pipeline for intelligent document processing using unstructured data, ensemble retrieval, and deep learning.

## Features

- 📂 PDF document processing with metadata extraction(page numbers, can be used o extract tables as well)
- 🔍 Hybrid retrieval (BM25 + Chroma DB)
- 🤖 Local LLM integration (Hugging Face)
- � Configurable through `config.py`

## Installation

1. Clone the repository:
```bash

pip install -r requirements.txt
```

## How it works 
```python
from query_processor import QueryProcessor
from document_processor import process_pdf
from retriever_setup import setup_retrievers

# Process all documents
all_docs = []
all_elements = []
for path in Config.DOCUMENT_PATHS:
    docs, elements = process_pdf(path)
    all_docs.extend(docs)
    all_elements.extend(elements)

# Build hybrid retriever
retriever = setup_retrievers(all_docs, all_elements)
qp = QueryProcessor(retriever)

# Simple query
response = qp.query("How do I troubleshoot a server crash?")
print(response)

# With context references
detailed_response = qp.query("Explain blade server maintenance with sources")
print(detailed_response)
