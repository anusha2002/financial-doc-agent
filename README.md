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

## Running the evaluation
1. Edit `eval_questions.csv` add your questions to this document
2. Run `python run_eval.py` — this runs every question through the agent and saves
   answers to `eval_results.csv`.
3. Open `eval_results.csv`, read each answer, and fill in the `correct` column with
   `y` or `n`. Add notes on *why* something failed in the `notes` column. This feature helps check accuracy.
4. Run `python score_eval.py` to get overall accuracy and a breakdown by category
   (factual / calculation / unanswerable / ambiguous).

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

