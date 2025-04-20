# query_processor.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFaceHub
from config import Config
from typing import List
from langchain_core.documents import Document

class QueryProcessor:
    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = self._setup_llm()
        self.prompt_template = ChatPromptTemplate.from_template("""
            You are an IT support specialist. Answer the question using only the provided context.
            If you don't know the answer, say "I couldn't find that information in the documentation."

            Context: {context}

            Question: {question}
            
            Answer in markdown format with clear sections if needed:
            """)

    def _setup_llm(self):
        """Configure local LLM (HuggingFace) instead of OpenAI"""
        return HuggingFaceHub(
            repo_id=Config.LLM_MODEL,
            model_kwargs={"temperature": 0.5, "max_length": 1024}
        )

    def format_context(self, docs: List[Document]) -> str:
        """Format retrieved documents with source metadata"""
        context_str = []
        for doc in docs:
            source = doc.metadata.get('source', 'unknown source')
            page = doc.metadata.get('page', 'N/A')
            context_str.append(
                f"Content: {doc.page_content}\n"
                f"Source: {source} (Page {page})\n"
                f"{'-'*40}"
            )
        return "\n\n".join(context_str)

    def query(self, question: str) -> str:
        """End-to-end query processing"""
        # Retrieve relevant documents
        context_docs = self.retriever.invoke(question)
        
        # Format context with metadata
        context_str = self.format_context(context_docs)
        
        # Generate response
        chain = self.prompt_template | self.llm
        response = chain.invoke({
            "context": context_str, 
            "question": question
        })
        
        return response

# Example usage
if __name__ == "__main__":
    from retriever_setup import setup_retrievers
    from document_processor import process_pdf
    
    # Initialize system
    docs, elements = process_pdf(Config.DOCUMENT_PATHS[0])
    retriever = setup_retrievers(docs, elements)
    qp = QueryProcessor(retriever)
    
    # Process query
    query = "Blade server description"
    response = qp.query(query)
    print(f"Question: {query}")
    print(f"Response:\n{response}")
