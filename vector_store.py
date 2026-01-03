"""
Vector Store using ChromaDB.

WHAT IS A VECTOR DATABASE?
==========================
A vector database stores embeddings and enables similarity search.
It answers: "Given this query vector, what are the most similar stored vectors?"

HOW CHROMADB WORKS:
===================
1. Create a "collection" (like a table in SQL)
2. Add documents with their embeddings
3. Query by embedding to find similar documents

Under the hood, ChromaDB uses algorithms like HNSW (Hierarchical Navigable 
Small World) for fast approximate nearest neighbor search.

WHY NOT JUST USE A LIST?
========================
You could store vectors in a Python list and loop through to find similar ones.
But that's O(n) - slow for large datasets. Vector DBs use indexing for O(log n).
"""

import chromadb
from chromadb.config import Settings


class VectorStore:
    """
    A simple wrapper around ChromaDB for educational purposes.
    
    This shows the essential operations:
    - Create/load a collection
    - Add documents with embeddings
    - Search for similar documents
    """
    
    def __init__(self, collection_name: str = "rag_documents", persist_dir: str = "./chroma_db"):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name for this collection of documents
            persist_dir: Directory to save the database (persists between runs)
        """
        # Create a persistent ChromaDB client
        # This saves data to disk so you don't lose it when the program ends
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        # Get or create a collection
        # Collections are like tables - they hold related documents
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "RAG document store"}
        )
        
        print(f"Initialized vector store: {collection_name}")
        print(f"  Persistence: {persist_dir}")
        print(f"  Current document count: {self.collection.count()}")
    
    def add_documents(self, documents: list[dict], embeddings: list[list[float]]):
        """
        Add documents with their embeddings to the store.
        
        Args:
            documents: List of dicts with 'content', 'source', etc.
            embeddings: Corresponding embedding vectors
        
        ChromaDB requires:
        - ids: Unique identifier for each document
        - embeddings: The vector representations
        - documents: The actual text content
        - metadatas: Additional info (source, chunk_index, etc.)
        """
        if not documents:
            return
        
        # Prepare data for ChromaDB
        ids = []
        texts = []
        metadatas = []
        
        # Get current count to generate unique IDs
        start_id = self.collection.count()
        
        for i, doc in enumerate(documents):
            ids.append(f"doc_{start_id + i}")
            texts.append(doc["content"])
            metadatas.append({
                "source": doc.get("source", "unknown"),
                "chunk_index": doc.get("chunk_index", 0)
            })
        
        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        
        print(f"Added {len(documents)} documents to vector store")
    
    def search(self, query_embedding: list[float], top_k: int = 3) -> list[dict]:
        """
        Find the most similar documents to the query.
        
        This is the core retrieval operation in RAG!
        
        Args:
            query_embedding: Vector representation of the query
            top_k: Number of results to return
        
        Returns:
            List of dicts with 'content', 'source', 'distance'
        
        HOW SIMILARITY SEARCH WORKS:
        ============================
        ChromaDB computes the distance between query_embedding and
        every stored embedding, then returns the closest ones.
        
        Default distance metric is L2 (Euclidean distance).
        Lower distance = more similar.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted = []
        for i in range(len(results["documents"][0])):
            formatted.append({
                "content": results["documents"][0][i],
                "source": results["metadatas"][0][i].get("source", "unknown"),
                "distance": results["distances"][0][i]
            })
        
        return formatted
    
    def clear(self):
        """Delete all documents from the collection."""
        # Delete the collection and recreate it
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name="rag_documents",
            metadata={"description": "RAG document store"}
        )
        print("Cleared all documents from vector store")
    
    def count(self) -> int:
        """Return the number of documents in the store."""
        return self.collection.count()


# =============================================================================
# DEMO: Run this file to see vector store operations
# =============================================================================
if __name__ == "__main__":
    from embeddings import get_embeddings_batch
    
    print("=" * 60)
    print("VECTOR STORE DEMO")
    print("=" * 60)
    
    # Create vector store
    store = VectorStore(collection_name="demo_collection", persist_dir="./chroma_db_demo")
    
    # Sample documents
    docs = [
        {"content": "Python is a programming language", "source": "doc1.txt"},
        {"content": "Cats are popular pets", "source": "doc2.txt"},
        {"content": "JavaScript runs in browsers", "source": "doc3.txt"},
    ]
    
    # Get embeddings
    print("\nGetting embeddings...")
    texts = [d["content"] for d in docs]
    embeddings = get_embeddings_batch(texts)
    
    # Add to store
    store.clear()  # Start fresh for demo
    store.add_documents(docs, embeddings)
    
    # Search
    print("\nSearching for: 'programming languages'")
    from embeddings import get_embedding
    query_emb = get_embedding("programming languages")
    results = store.search(query_emb, top_k=2)
    
    print("\nSearch results:")
    for r in results:
        print(f"  - {r['content']} (distance: {r['distance']:.4f})")
