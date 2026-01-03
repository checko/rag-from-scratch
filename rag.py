"""
RAG (Retrieval-Augmented Generation) Pipeline.

THE COMPLETE RAG FLOW:
======================

1. INDEXING (done once):
   Documents -> Chunks -> Embeddings -> Vector Store

2. QUERYING (for each question):
   Question -> Embed -> Search Vector Store -> Get similar chunks
   -> Build prompt with context -> Call LLM -> Answer

WHY RAG WORKS:
==============
LLMs have knowledge cutoffs and can't access your private data.
RAG solves this by:
1. Finding relevant information from YOUR documents
2. Including it in the prompt as context
3. Letting the LLM generate answers based on that context

This is like giving someone a relevant book page before asking a question,
rather than expecting them to know everything from memory.
"""

import requests
from config import OLLAMA_BASE_URL, LLM_MODEL, TOP_K
from chunker import chunk_documents
from embeddings import get_embedding, get_embeddings_batch
from vector_store import VectorStore


class RAG:
    """
    A complete RAG pipeline.
    
    This class ties together all the components:
    - Document chunking
    - Embedding generation
    - Vector storage
    - LLM generation
    """
    
    def __init__(self, collection_name: str = "rag_documents"):
        """Initialize RAG with a vector store."""
        self.vector_store = VectorStore(collection_name=collection_name)
    
    def index_documents(self, documents: list[dict]):
        """
        Index documents into the vector store.
        
        This is the "ingestion" phase of RAG:
        1. Split documents into chunks
        2. Generate embeddings for each chunk
        3. Store chunks with embeddings in vector database
        
        Args:
            documents: List of dicts with 'content' and 'source' keys
        """
        print("\n" + "=" * 60)
        print("INDEXING DOCUMENTS")
        print("=" * 60)
        
        # Step 1: Chunk documents
        print("\n1. Chunking documents...")
        chunks = chunk_documents(documents)
        print(f"   Created {len(chunks)} chunks from {len(documents)} documents")
        
        # Step 2: Generate embeddings
        print("\n2. Generating embeddings...")
        texts = [chunk["content"] for chunk in chunks]
        embeddings = get_embeddings_batch(texts)
        print(f"   Generated {len(embeddings)} embeddings (dim={len(embeddings[0])})")
        
        # Step 3: Store in vector database
        print("\n3. Storing in vector database...")
        self.vector_store.add_documents(chunks, embeddings)
        print(f"   Total documents in store: {self.vector_store.count()}")
        
        print("\nIndexing complete!")
    
    def query(self, question: str, show_context: bool = True) -> str:
        """
        Answer a question using RAG.
        
        THE RAG QUERY PIPELINE:
        1. Embed the question (convert to vector)
        2. Search for similar chunks in vector store
        3. Build a prompt with retrieved context
        4. Call the LLM to generate an answer
        
        Args:
            question: The user's question
            show_context: Whether to print retrieved context (for learning)
        
        Returns:
            The LLM's answer
        """
        print("\n" + "=" * 60)
        print(f"QUERY: {question}")
        print("=" * 60)
        
        # Step 1: Embed the question
        print("\n1. Embedding question...")
        query_embedding = get_embedding(question)
        print(f"   Embedding dimension: {len(query_embedding)}")
        
        # Step 2: Retrieve similar chunks
        print(f"\n2. Retrieving top {TOP_K} similar chunks...")
        results = self.vector_store.search(query_embedding, top_k=TOP_K)
        
        if show_context:
            print("\n   Retrieved context:")
            for i, r in enumerate(results):
                print(f"\n   --- Chunk {i+1} (distance: {r['distance']:.4f}) ---")
                print(f"   Source: {r['source']}")
                preview = r['content'][:200] + "..." if len(r['content']) > 200 else r['content']
                print(f"   Content: {preview}")
        
        # Step 3: Build the augmented prompt
        print("\n3. Building augmented prompt...")
        context = "\n\n".join([r["content"] for r in results])
        
        # This prompt format is crucial for RAG!
        # We explicitly tell the LLM to use the provided context.
        prompt = f"""Answer the question based on the context provided below.
If the context doesn't contain enough information to answer, say so.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
        
        if show_context:
            print(f"   Prompt length: {len(prompt)} characters")
        
        # Step 4: Call the LLM
        print("\n4. Calling LLM for answer...")
        answer = self._call_llm(prompt)
        
        print("\n" + "=" * 60)
        print("ANSWER:")
        print("=" * 60)
        print(answer)
        
        return answer
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call Ollama LLM to generate a response.
        
        This is a direct HTTP call to Ollama's API.
        No LangChain or other abstractions!
        
        API Details:
            Endpoint: POST /api/generate
            Body: {"model": "...", "prompt": "...", "stream": false}
        """
        url = f"{OLLAMA_BASE_URL}/api/generate"
        
        payload = {
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False  # Get complete response at once
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        return result["response"]
    
    def clear(self):
        """Clear all indexed documents."""
        self.vector_store.clear()


# =============================================================================
# DEMO: Run this file to see the complete RAG pipeline
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("RAG PIPELINE DEMO")
    print("=" * 60)
    
    # Create RAG instance
    rag = RAG(collection_name="demo_rag")
    rag.clear()  # Start fresh
    
    # Sample documents about a fictional company
    documents = [
        {
            "content": """
            Acme Corp was founded in 2020 by Jane Smith and John Doe.
            The company specializes in AI-powered productivity tools.
            Their flagship product is "TaskMaster", an intelligent task
            management system that uses machine learning to prioritize work.
            """,
            "source": "company_info.txt"
        },
        {
            "content": """
            TaskMaster pricing: 
            - Free tier: Up to 10 tasks, basic features
            - Pro tier: $9.99/month, unlimited tasks, AI prioritization
            - Enterprise: Custom pricing, SSO, dedicated support
            
            All plans include mobile apps for iOS and Android.
            """,
            "source": "pricing.txt"
        },
        {
            "content": """
            Acme Corp office locations:
            - Headquarters: San Francisco, CA
            - Engineering: Austin, TX  
            - Sales: New York, NY
            
            The company has 150 employees as of 2024.
            """,
            "source": "locations.txt"
        }
    ]
    
    # Index documents
    rag.index_documents(documents)
    
    # Ask questions
    print("\n" + "#" * 60)
    print("ASKING QUESTIONS")
    print("#" * 60)
    
    questions = [
        "Who founded Acme Corp?",
        "How much does the Pro tier cost?",
        "Where is the engineering office located?"
    ]
    
    for q in questions:
        rag.query(q)
        print("\n")
