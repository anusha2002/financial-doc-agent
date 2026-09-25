import chromadb
from sentence_transformers import SentenceTransformer
import ollama

DB_DIR = "chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.1"   # swap for any model you've pulled, e.g. "mistral", "qwen2.5"
TOP_K = 8

_embed_model = None
_collection = None


def _get_resources():
    global _embed_model, _collection
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
        client = chromadb.PersistentClient(path=DB_DIR)
        _collection = client.get_collection("financial_docs")
    return _embed_model, _collection


def retrieve(query: str, top_k: int = TOP_K):
    """Return list of {text, source, page, score} for the most relevant chunks."""
    model, collection = _get_resources()
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    hits = []
    for text, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({"text": text, "source": meta["source"], "page": meta["page"], "score": dist})
    return hits


def answer_with_citations(query: str, top_k: int = TOP_K) -> str:
    """Retrieve context, then ask the local model to answer ONLY from that context, with citations."""
    hits = retrieve(query, top_k=top_k)
    if not hits:
        return "No relevant documents were indexed. Run ingest.py first."

    context_block = "\n\n".join(
        f"[Source: {h['source']}, page {h['page']}]\n{h['text']}" for h in hits
    )

    system_prompt = (
        "You are a financial document assistant. Answer the user's question using ONLY "
        "the provided context. If the context does not contain enough information to answer, "
        "say so explicitly rather than guessing. Every claim you make must cite the source "
        "document and page number it came from, like this: (Source: filename.pdf, p.3)."
    )

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {query}"},
        ],
    )
    return response["message"]["content"]


if __name__ == "__main__":
    q = input("Ask a question about your documents: ")
    print("\n" + answer_with_citations(q))