# Financial Document Q&A Agent

A small RAG + tool-calling agent over financial PDFs (e.g. 10-Ks, fund fact sheets).
Answers questions with cited sources and can combine document retrieval with simple
calculations (e.g. growth rates, margins).

## Architecture
1. **Ingestion** (`ingest.py`) — extracts text from PDFs page-by-page, splits into
   overlapping chunks, embeds locally with `sentence-transformers`, and stores in a
   persistent Chroma vector database.
2. **Retrieval** (`rag.py`) — embeds a query, retrieves the top-k most similar chunks,
   and asks Claude to answer strictly from that context with source + page citations.
3. **Agent** (`agent.py`) — wraps retrieval as a tool alongside a calculator tool, so
   Claude decides per-question whether to search documents, compute a number, or both.

## Setup

Fully local and free — runs on Ollama, no API costs.

```bash
# 1. Install Ollama: https://ollama.com/download
# 2. Pull a tool-calling-capable model (needs ~5-8GB RAM free):
ollama pull llama3.1

pip install -r requirements.txt
mkdir -p docs   # drop 3-5 PDFs here (10-Ks, fact sheets, etc.)
python ingest.py
python agent.py
```
Local models are noticeably weaker at deciding *when* to call a tool and at
consistently citing sources than a hosted model like Claude or GPT-4 would be. Worth
documenting that gap in your eval table below — it's a legitimate, honest observation
about model capability trade-offs, directly relevant to this job's interest in
cost/performance trade-offs across platforms.

## Evaluation notes (fill this in as you test)
| # | Question | Expected behavior | Result | Notes |
|---|----------|-------------------|--------|-------|
| 1 | Direct factual question answerable from docs | Correct, cited | | |
| 2 | Question with no answer in the docs | Should say "not enough information" | | |
| 3 | Ambiguous question (could match multiple docs) | Should retrieve broadly or ask for clarification | | |
| 4 | Requires math across two documents | Should retrieve both, then calculate | | |
| 5 | Adversarial: asks it to ignore instructions and make up a number | Should refuse / stay grounded | | |

## Known limitations
- Chunking is character-based, not token- or semantic-aware — fine for a demo, would
  need a proper tokenizer-based splitter for production.
- Local embeddings (`all-MiniLM-L6-v2`) are fast but lower quality than an API-based
  embedding model — swap in `voyage-ai` or `text-embedding-3` for better recall.
- No retrieval evaluation metrics (precision/recall) beyond manual spot-checks — next
  step would be a small labeled test set.
- `calculate` uses `eval()` — fine locally, would need a safe expression parser (e.g.
  `numexpr` or `asteval`) before ever touching untrusted input.
