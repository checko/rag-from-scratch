# rag-from-scratch

> Learn how RAG works by building it from scratch with Ollama and ChromaDB — no LangChain!


## What is RAG?

RAG combines **retrieval** (finding relevant documents) with **generation** (LLM answering questions). Instead of relying only on the LLM's training data, we give it relevant context at query time.

## How RAG Works

### Vector Database Structure

Each record in the vector database contains:
- **Original text chunk** (the actual content)
- **Embedding vector** (numerical representation for similarity search)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         VECTOR DATABASE                             │
│                                                                     │
│  Record 1: { chunk: "Paris is capital...", embedding: [0.2, 0.8...]}│
│  Record 2: { chunk: "Eiffel Tower is...",  embedding: [0.3, 0.7...]}│
│  Record 3: { chunk: "Python is a...",      embedding: [-0.5, 0.1...]}│
└─────────────────────────────────────────────────────────────────────┘
```

### Query Flow

```
User Question: "What is the capital of France?"
        │
        ▼
┌───────────────────┐
│ 1. Embed Question │  →  [0.25, 0.75, ...]
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ 2. Vector Search  │  →  Find records with similar embeddings
└───────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────┐
│ 3. Retrieved Chunks (original text, NOT vectors) │
│    - "Paris is the capital of France..."         │
│    - "France is located in Western Europe..."    │
└───────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Build Augmented Prompt                                   │
│                                                             │
│    "Answer based on this context:                           │
│                                                             │
│     Context: Paris is the capital of France...              │
│                                                             │
│     Question: What is the capital of France?                │
│                                                             │
│     Answer:"                                                │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────┐
│ 5. LLM Generates  │  →  "Paris"
└───────────────────┘
```

**Key insight**: Embeddings are only used for **finding** similar records. The LLM never sees vectors—it only sees the **original text chunks** as context!

## Project Structure

```
├── config.py        # Ollama settings (models, endpoints)
├── chunker.py       # Document → overlapping text chunks
├── embeddings.py    # Text → 768-dim vectors via Ollama
├── vector_store.py  # ChromaDB add/search operations
├── rag.py           # Complete RAG pipeline
└── demo.py          # Interactive Q&A demo
```

## Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Make sure Ollama is running with required models
ollama list  # Should show deepseek-r1 and nomic-embed-text
```

## Usage

```bash
source .venv/bin/activate
python demo.py
```

Example questions:
- "What is RAG?"
- "What are the components of a RAG system?"
- "When should I use RAG vs fine-tuning?"

## Requirements

- Python 3.6+
- Ollama with `deepseek-r1` and `nomic-embed-text` models
- ChromaDB (installed via pip)

## Components Explained

| Component | File | Purpose |
|-----------|------|---------|
| **Chunker** | `chunker.py` | Splits documents into 500-char chunks with 50-char overlap |
| **Embeddings** | `embeddings.py` | Calls Ollama API to convert text → 768-dim vectors |
| **Vector Store** | `vector_store.py` | ChromaDB wrapper for storing/searching embeddings |
| **RAG Pipeline** | `rag.py` | Orchestrates: index documents → query → generate answer |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

