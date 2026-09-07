"""Vector store setup: local Chroma DB with sentence-transformers embeddings,
plus a cross-encoder re-ranking stage on top of raw vector similarity.

Vector search (cosine distance over embeddings) is fast but approximate: it's
good at finding candidates that are topically related, but not always the
best judge of which candidate actually answers the question. A cross-encoder
re-ranker looks at the (question, chunk) pair *together* rather than as two
independent vectors, which makes it slower but noticeably more precise — so
the pattern here is "cast a wide net with vector search, then have the
re-ranker pick the best few from that net."
"""
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "documents"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATE_POOL = 15  # how many vector-search results to feed the re-ranker

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

_client = chromadb.PersistentClient(path=CHROMA_PATH)
_cross_encoder = None


def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(RERANK_MODEL)
    return _cross_encoder


def get_collection():
    return _client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )


def query(text: str, n_results: int = 4, rerank: bool = True) -> list[dict]:
    collection = get_collection()
    if collection.count() == 0:
        return []

    pool_size = min(CANDIDATE_POOL if rerank else n_results, collection.count())
    results = collection.query(query_texts=[text], n_results=pool_size)

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({"text": doc, "source": meta.get("source", "unknown"), "distance": dist})

    if rerank and hits:
        pairs = [(text, h["text"]) for h in hits]
        scores = get_cross_encoder().predict(pairs)
        for hit, score in zip(hits, scores):
            hit["rerank_score"] = float(score)
        hits.sort(key=lambda h: h["rerank_score"], reverse=True)

    return hits[:n_results]
