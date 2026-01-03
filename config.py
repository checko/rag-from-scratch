"""
Configuration for RAG system.

This file contains all the settings for connecting to Ollama and configuring
the RAG pipeline. Ollama runs locally, so we use localhost.
"""

# =============================================================================
# OLLAMA CONFIGURATION
# =============================================================================

# Ollama API endpoint (default local installation)
OLLAMA_BASE_URL = "http://localhost:11434"

# LLM model for generating answers
# deepseek-r1 is a powerful reasoning model good for Q&A
LLM_MODEL = "deepseek-r1:latest"

# Embedding model for converting text to vectors
# nomic-embed-text produces 768-dimensional embeddings
EMBEDDING_MODEL = "nomic-embed-text:latest"

# =============================================================================
# RAG CONFIGURATION
# =============================================================================

# Number of similar chunks to retrieve for context
TOP_K = 3

# =============================================================================
# CHUNKING CONFIGURATION
# =============================================================================

# Size of each text chunk in characters
CHUNK_SIZE = 500

# Overlap between chunks to preserve context across boundaries
CHUNK_OVERLAP = 50
