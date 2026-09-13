# Financial Document Q&A Agent

A small RAG + tool-calling agent over financial PDFs (e.g. 10-Ks, fund fact sheets).
This project answers questions with cited sources and can combine document retrieval with simple calculations (e.g. growth rates, margins).

## Architecture
1. **Ingestion** (`ingest.py`) — extracts text from PDFs page-by-page, splits into
   overlapping chunks, embeds locally with `sentence-transformers`, and stores in a
   persistent Chroma vector database.
2. **Retrieval** (`rag.py`) — embeds a query, retrieves the top-k most similar chunks,
   and asks the local LLM (via Ollama) to answer strictly from that context with source + page citations.
3. **Agent** (`agent.py`) — wraps retrieval as a tool alongside a calculator tool, so
   the model decides per-question whether to search documents, compute a number, or both.

## Setup

This project is fully local and free — runs on Ollama, no API costs.

```bash
# 1. Install Ollama: https://ollama.com/download
# 2. Pull a tool-calling-capable model (needs ~5-8GB RAM free):
ollama pull llama3.1

pip install -r requirements.txt
mkdir -p docs   # drop 3-5 PDFs here (10-Ks, fact sheets, etc.)
python ingest.py
python agent.py
```

