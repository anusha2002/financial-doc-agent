import os
import glob
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer

DOCS_DIR = "docs"
DB_DIR = "chroma_db"
CHUNK_SIZE = 800       # characters, not tokens -- simple and good enough for a weekend project
CHUNK_OVERLAP = 150
EMBED_MODEL = "all-MiniLM-L6-v2"  # small, fast, runs locally


def extract_text_by_page(pdf_path):
    """Return list of (page_number, text) tuples."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i + 1, text))
    return pages


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Simple sliding-window chunker."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def build_index():
    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=DB_DIR)
    # Fresh collection each run -- fine for a small project; switch to upsert logic if you scale this up.
    try:
        client.delete_collection("financial_docs")
    except Exception:
        pass
    collection = client.create_collection("financial_docs")

    pdf_paths = glob.glob(os.path.join(DOCS_DIR, "*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {DOCS_DIR}/. Add some and re-run.")
        return

    doc_id = 0
    all_chunks, all_metadatas, all_ids = [], [], []

    for pdf_path in pdf_paths:
        doc_name = os.path.basename(pdf_path)
        print(f"Processing {doc_name}...")
        pages = extract_text_by_page(pdf_path)
        for page_num, page_text in pages:
            for chunk in chunk_text(page_text):
                if len(chunk.strip()) < 50:
                    continue  # skip near-empty chunks
                all_chunks.append(chunk)
                all_metadatas.append({"source": doc_name, "page": page_num})
                all_ids.append(f"chunk_{doc_id}")
                doc_id += 1

    print(f"Embedding {len(all_chunks)} chunks...")
    embeddings = model.encode(all_chunks, show_progress_bar=True).tolist()

    collection.add(
        ids=all_ids,
        embeddings=embeddings,
        documents=all_chunks,
        metadatas=all_metadatas,
    )
    print(f"Indexed {len(all_chunks)} chunks from {len(pdf_paths)} document(s) into {DB_DIR}/")


if __name__ == "__main__":
    build_index()
