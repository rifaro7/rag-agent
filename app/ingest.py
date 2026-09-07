"""Chunk and embed everything in documents/ into the Chroma vector store.

Usage: python -m app.ingest
"""
import pathlib

from pypdf import PdfReader

from app.rag import get_collection

DOCS_DIR = pathlib.Path("documents")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def read_text(path: pathlib.Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(errors="ignore")


def chunk(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start = end - CHUNK_OVERLAP
    return [c.strip() for c in chunks if c.strip()]


def main():
    files = [p for p in DOCS_DIR.glob("**/*") if p.suffix.lower() in (".pdf", ".txt", ".md")]
    if not files:
        print(f"No .pdf/.txt/.md files found in {DOCS_DIR}/. Add some and re-run.")
        return

    collection = get_collection()
    ids, docs, metadatas = [], [], []
    for path in files:
        text = read_text(path)
        for i, piece in enumerate(chunk(text)):
            ids.append(f"{path.name}-{i}")
            docs.append(piece)
            metadatas.append({"source": path.name})

    if not docs:
        print("Found files but extracted no text from them.")
        return

    collection.upsert(ids=ids, documents=docs, metadatas=metadatas)
    print(f"Ingested {len(docs)} chunks from {len(files)} file(s) into '{collection.name}'.")


if __name__ == "__main__":
    main()
