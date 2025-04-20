def process_pdf(file_path: str) -> Tuple[List[Document], List[Document]]:
    """
    Extracts text/tables from PDFs and returns LangChain Documents.
    Args:
        file_path: Path to the PDF (e.g., "data/server_guide.pdf")
    Returns:
        Tuple of (full_page_docs, element_docs)
    """
    # Extract elements from PDF
    elements = partition_pdf(
        file_path,
        strategy="auto",
        infer_table_structure=True,
        include_page_breaks=True,
    )
    
    # Corrected filtering - use metadata. category_name instead of category
    filtered_elements = [
        el for el in elements 
        if (getattr(el.metadata, "category_name", None) not in {"Footer", "Image", "Header"}
          and el.text.strip()  # This checks for non-empty text after stripping whitespace
        )
    ]

    
    # Rest of the function remains the same...
    page_docs = []
    element_docs = []
    
    for page_num, page_elements in group_by_page(filtered_elements).items():
        page_text = "\n".join([el.text for el in page_elements])
        page_docs.append(
            Document(
                page_content=page_text,
                metadata={"source": file_path, "page": page_num}
            )
        )
        
        for el in page_elements:
            element_docs.append(
                Document(
                    page_content=el.text,
                    metadata={
                        "source": file_path,
                        "page": page_num,
                        # "type": el.metadata.category_name  # Updated here too
                    }
                )
            )
    
    return page_docs, element_docs

def group_by_page(elements):
    """Helper to group elements by page number with None handling."""
    pages = {}
    for el in elements:
        page_num = getattr(el.metadata, 'page_number', None)
        if page_num is None:
            page_num = "unknown"  # Or use 0, -1, or any other placeholder
        pages.setdefault(page_num, []).append(el)
    return pages
