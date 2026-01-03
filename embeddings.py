"""
Embedding Generation using Ollama.

WHAT ARE EMBEDDINGS?
====================
Embeddings convert text into numerical vectors (lists of numbers).
Similar texts produce similar vectors, enabling semantic search.

Example:
    "cat" -> [0.2, 0.8, -0.1, ...]
    "dog" -> [0.3, 0.7, -0.2, ...]  <- similar to cat
    "car" -> [-0.5, 0.1, 0.9, ...]  <- different from cat

HOW OLLAMA EMBEDDINGS WORK:
===========================
1. We send text to Ollama's /api/embed endpoint
2. Ollama uses the nomic-embed-text model to process the text
3. It returns a 768-dimensional vector representing the text's meaning

The vector captures semantic meaning, not just keywords.
"The cat sat on the mat" and "A feline rested on the rug" 
would have similar embeddings despite different words.
"""

import requests
from config import OLLAMA_BASE_URL, EMBEDDING_MODEL


def get_embedding(text: str) -> list[float]:
    """
    Get embedding vector for a single text using Ollama API.
    
    This is the core function - it shows exactly how to call
    an embedding API without any abstraction layers.
    
    Args:
        text: The text to embed
    
    Returns:
        List of floats representing the embedding vector
    
    API Details:
        Endpoint: POST /api/embed
        Body: {"model": "nomic-embed-text", "input": "text"}
        Response: {"embeddings": [[0.1, 0.2, ...]]}
    """
    # Ollama's embedding endpoint
    url = f"{OLLAMA_BASE_URL}/api/embed"
    
    # Request body - very simple!
    payload = {
        "model": EMBEDDING_MODEL,
        "input": text
    }
    
    # Make the HTTP request
    response = requests.post(url, json=payload)
    response.raise_for_status()  # Raise error if request failed
    
    # Extract the embedding from response
    # Ollama returns {"embeddings": [[vector]]} for single input
    result = response.json()
    embedding = result["embeddings"][0]
    
    return embedding


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Get embeddings for multiple texts efficiently.
    
    Ollama supports batch embedding - sending multiple texts at once
    is more efficient than calling get_embedding() in a loop.
    
    Args:
        texts: List of texts to embed
    
    Returns:
        List of embedding vectors (one per input text)
    """
    if not texts:
        return []
    
    url = f"{OLLAMA_BASE_URL}/api/embed"
    
    # Send all texts at once
    payload = {
        "model": EMBEDDING_MODEL,
        "input": texts  # List of strings
    }
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    result = response.json()
    return result["embeddings"]


# =============================================================================
# DEMO: Run this file to see embeddings in action
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("EMBEDDING DEMO")
    print("=" * 60)
    
    # Test texts
    texts = [
        "The cat sat on the mat",
        "A feline rested on the rug",  # Semantically similar to first
        "Python is a programming language"  # Different topic
    ]
    
    print("Getting embeddings for:")
    for i, text in enumerate(texts):
        print(f"  {i}: {text}")
    print()
    
    # Get embeddings
    embeddings = get_embeddings_batch(texts)
    
    print(f"Embedding dimension: {len(embeddings[0])}")
    print()
    
    # Show first few values of each embedding
    for i, (text, emb) in enumerate(zip(texts, embeddings)):
        print(f"Text {i}: {text[:30]}...")
        print(f"  Vector (first 5 values): {emb[:5]}")
        print()
    
    # Calculate similarity (cosine similarity)
    import math
    
    def cosine_similarity(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b)
    
    print("Similarity scores (cosine similarity):")
    print(f"  cat/feline: {cosine_similarity(embeddings[0], embeddings[1]):.4f}")
    print(f"  cat/python: {cosine_similarity(embeddings[0], embeddings[2]):.4f}")
    print()
    print("Notice: Similar meanings = higher similarity score!")
