"""
Document Chunker for RAG.

WHY CHUNKING?
=============
1. LLMs have context limits - we can't pass entire documents
2. Embeddings work better on focused, coherent text pieces
3. Smaller chunks = more precise retrieval

HOW IT WORKS:
=============
We split text into overlapping chunks:

    [----chunk 1----]
              [----chunk 2----]
                        [----chunk 3----]
    
The overlap ensures we don't lose context at chunk boundaries.
For example, if a sentence spans two chunks, the overlap preserves it.
"""

from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: The input text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Number of overlapping characters between chunks
    
    Returns:
        List of text chunks
    
    Example:
        >>> text = "Hello world. This is a test document."
        >>> chunks = chunk_text(text, chunk_size=20, overlap=5)
        >>> print(chunks)
        ['Hello world. This i', 'his is a test docum', 'ocument.']
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Get chunk from start to start + chunk_size
        end = start + chunk_size
        chunk = text[start:end]
        
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)
        
        # Move start position, accounting for overlap
        # If overlap is 50 and chunk_size is 500, we move 450 characters forward
        start += chunk_size - overlap
    
    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk multiple documents, preserving metadata.
    
    Args:
        documents: List of dicts with 'content' and optional 'source' keys
    
    Returns:
        List of chunks with metadata (source, chunk_index)
    
    Example:
        >>> docs = [{"content": "Long text...", "source": "file.txt"}]
        >>> chunks = chunk_documents(docs)
        >>> print(chunks[0])
        {'content': 'Long tex...', 'source': 'file.txt', 'chunk_index': 0}
    """
    all_chunks = []
    
    for doc in documents:
        content = doc.get("content", "")
        source = doc.get("source", "unknown")
        
        # Split this document into chunks
        text_chunks = chunk_text(content)
        
        # Create chunk objects with metadata
        for i, chunk in enumerate(text_chunks):
            all_chunks.append({
                "content": chunk,
                "source": source,
                "chunk_index": i
            })
    
    return all_chunks


# =============================================================================
# DEMO: Run this file directly to see chunking in action
# =============================================================================
if __name__ == "__main__":
    sample_text = """
    Retrieval-Augmented Generation (RAG) is a technique that combines 
    information retrieval with text generation. It works by first finding 
    relevant documents from a knowledge base, then using those documents 
    as context for a language model to generate accurate answers.
    
    The key components of RAG are:
    1. A document store (vector database)
    2. An embedding model to convert text to vectors
    3. A retrieval system to find similar documents
    4. A language model to generate responses
    """
    
    print("=" * 60)
    print("CHUNKING DEMO")
    print("=" * 60)
    print(f"Original text length: {len(sample_text)} characters")
    print(f"Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")
    print()
    
    chunks = chunk_text(sample_text)
    print(f"Number of chunks: {len(chunks)}")
    print()
    
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i} ({len(chunk)} chars) ---")
        print(chunk[:100] + "..." if len(chunk) > 100 else chunk)
        print()
