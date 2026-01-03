"""
Interactive RAG Demo.

This script demonstrates the complete RAG workflow:
1. Load sample documents
2. Chunk and embed them
3. Store in vector database
4. Answer questions using retrieved context

Run this to see RAG in action!
"""

from rag import RAG


def load_sample_documents():
    """Load sample documents for the demo."""
    return [
        {
            "content": """
            # What is RAG?
            
            Retrieval-Augmented Generation (RAG) is a technique that enhances 
            Large Language Models by giving them access to external knowledge.
            
            Traditional LLMs can only use information from their training data,
            which has a cutoff date and doesn't include private data. RAG solves
            this by retrieving relevant documents at query time and including
            them in the prompt.
            
            The key insight is: instead of fine-tuning a model on new data 
            (expensive, slow), we can just show it relevant documents when needed.
            """,
            "source": "rag_basics.md"
        },
        {
            "content": """
            # Components of a RAG System
            
            1. **Document Loader**: Reads documents from files, databases, APIs
            
            2. **Text Splitter/Chunker**: Breaks documents into smaller pieces
               - Why? Embeddings work better on focused text
               - Typical chunk size: 200-1000 characters
               - Overlap prevents losing context at boundaries
            
            3. **Embedding Model**: Converts text to vectors
               - Popular: OpenAI ada-002, nomic-embed-text, sentence-transformers
               - Output: 384-1536 dimensional vectors
            
            4. **Vector Database**: Stores and searches embeddings
               - Examples: ChromaDB, Pinecone, Weaviate, FAISS
               - Enables fast similarity search (not just keyword matching)
            
            5. **LLM**: Generates answers using retrieved context
               - Any chat model works: GPT-4, Claude, Llama, etc.
            """,
            "source": "rag_components.md"
        },
        {
            "content": """
            # Vector Similarity Search
            
            How does a vector database find similar documents?
            
            1. **Embedding**: Text is converted to a high-dimensional vector
               "cat" -> [0.2, 0.8, -0.1, 0.5, ...]
            
            2. **Distance Metrics**: Common methods to measure similarity:
               - Cosine similarity: angle between vectors (most common)
               - Euclidean distance: straight-line distance
               - Dot product: magnitude-aware similarity
            
            3. **Approximate Nearest Neighbor (ANN)**: 
               For large datasets, exact search is too slow.
               Algorithms like HNSW build graph indexes for fast approximate search.
               Trade-off: slight accuracy loss for huge speed gains.
            
            4. **Result**: Top-K most similar documents returned
            """,
            "source": "vector_search.md"
        },
        {
            "content": """
            # RAG vs Fine-tuning
            
            When to use RAG:
            - Need access to frequently updated information
            - Want to cite sources / show provenance
            - Limited compute budget
            - Need to work with private/proprietary data
            
            When to use Fine-tuning:
            - Need to change model behavior/style
            - Want consistent, specialized responses
            - Have lots of training examples
            - Don't need real-time updates
            
            In practice, many systems combine both:
            Fine-tune for style/behavior + RAG for knowledge.
            """,
            "source": "rag_vs_finetuning.md"
        },
        {
            "content": """
            # Common RAG Challenges
            
            1. **Chunking Strategy**: 
               Too small = lose context
               Too large = noisy retrieval
               Solution: experiment with sizes, use semantic chunking
            
            2. **Retrieval Quality**:
               Wrong documents = wrong answers
               Solutions: hybrid search (keyword + semantic), reranking
            
            3. **Context Window Limits**:
               Can't fit all retrieved docs in prompt
               Solutions: summarization, selective inclusion
            
            4. **Hallucination**:
               LLM may ignore context or make things up
               Solutions: prompt engineering, citation requirements
            
            5. **Latency**:
               Embedding + search + LLM adds latency
               Solutions: caching, async processing, smaller models
            """,
            "source": "rag_challenges.md"
        }
    ]


def main():
    print("=" * 60)
    print("RAG DEMO - Learn How RAG Works!")
    print("=" * 60)
    
    # Initialize RAG
    rag = RAG(collection_name="rag_demo")
    
    # Check if we need to index documents
    if rag.vector_store.count() == 0:
        print("\nNo documents indexed. Loading sample documents...")
        documents = load_sample_documents()
        rag.index_documents(documents)
    else:
        print(f"\nFound {rag.vector_store.count()} existing documents in index.")
        reindex = input("Re-index documents? (y/n): ").strip().lower()
        if reindex == 'y':
            rag.clear()
            documents = load_sample_documents()
            rag.index_documents(documents)
    
    # Interactive Q&A loop
    print("\n" + "=" * 60)
    print("INTERACTIVE Q&A")
    print("=" * 60)
    print("Ask questions about RAG! Type 'quit' to exit.")
    print("Example questions:")
    print("  - What is RAG?")
    print("  - What are the components of a RAG system?")
    print("  - When should I use RAG vs fine-tuning?")
    print("  - What are common challenges in RAG?")
    
    while True:
        print("\n" + "-" * 40)
        question = input("Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not question:
            continue
        
        rag.query(question)


if __name__ == "__main__":
    main()
