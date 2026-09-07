"""Measures retrieval quality on a small, hand-written question set — and
compares plain vector search against vector search + cross-encoder re-ranking.

The corpus deliberately includes near-duplicate "distractor" documents (two
API docs, two leave policies, two onboarding checklists) that share a lot of
vocabulary but differ in the specific facts. That's what makes this a real
test: pure embedding similarity often can't tell "the Nimbus API" from "the
Comet API" apart, since they're topically almost identical. A cross-encoder,
which reads the question and the candidate passage together instead of
comparing two independent vectors, is meant to do better at exactly that.

Recall@4 alone tends to saturate on a corpus this small (there just aren't
enough chunks for the right one to fall outside the top 4), so the real
signal here is Mean Reciprocal Rank: how close to #1 the correct chunk lands.

Uses its own ephemeral (in-memory) Chroma collection over eval/corpus/, so it
never touches the real chroma_db/ your documents live in.

Usage: python -m eval.run_eval
"""
import json
import pathlib

import chromadb

from app.ingest import chunk
from app.rag import embedding_fn, get_cross_encoder

CORPUS_DIR = pathlib.Path(__file__).parent / "corpus"
QUESTIONS_FILE = pathlib.Path(__file__).parent / "questions.json"
TOP_K = 4
CANDIDATE_POOL = 15


def build_eval_collection():
    client = chromadb.EphemeralClient()
    collection = client.create_collection(name="eval", embedding_function=embedding_fn)

    ids, docs, metadatas = [], [], []
    for path in sorted(CORPUS_DIR.glob("*.txt")):
        for i, piece in enumerate(chunk(path.read_text())):
            ids.append(f"{path.name}-{i}")
            docs.append(piece)
            metadatas.append({"source": path.name})
    collection.upsert(ids=ids, documents=docs, metadatas=metadatas)
    return collection


def ranked_baseline(collection, question: str) -> list[dict]:
    """Full candidate pool, ordered by vector distance (closest first)."""
    pool = min(CANDIDATE_POOL, collection.count())
    results = collection.query(query_texts=[question], n_results=pool)
    return [
        {"text": doc, "source": meta["source"]}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]


def ranked_reranked(collection, question: str, baseline_pool: list[dict]) -> list[dict]:
    """Same candidate pool, re-ordered by cross-encoder relevance score."""
    pairs = [(question, h["text"]) for h in baseline_pool]
    scores = get_cross_encoder().predict(pairs)
    scored = [dict(h, score=float(s)) for h, s in zip(baseline_pool, scores)]
    scored.sort(key=lambda h: h["score"], reverse=True)
    return scored


def find_rank(ranked_hits: list[dict], expected_source: str, expected_keywords: list[str]) -> int | None:
    """1-indexed position of the first chunk that's both the right document
    and actually contains the answer (not just any chunk from that file)."""
    for i, h in enumerate(ranked_hits, start=1):
        if h["source"] == expected_source and all(
            kw.lower() in h["text"].lower() for kw in expected_keywords
        ):
            return i
    return None


def main():
    questions = json.loads(QUESTIONS_FILE.read_text())
    collection = build_eval_collection()

    rows = []
    for q in questions:
        baseline_pool = ranked_baseline(collection, q["question"])
        reranked_pool = ranked_reranked(collection, q["question"], baseline_pool)

        b_rank = find_rank(baseline_pool, q["expected_source"], q["expected_keywords"])
        r_rank = find_rank(reranked_pool, q["expected_source"], q["expected_keywords"])
        rows.append((q["question"], b_rank, r_rank))

    n = len(questions)
    print(f"{'question':<62} {'baseline':>10} {'reranked':>10}")
    print("-" * 84)

    def fmt(rank):
        if rank is None:
            return "not found"
        return f"#{rank}" + (" (top-4)" if rank <= TOP_K else "")

    b_recall = r_recall = 0
    b_rr_sum = r_rr_sum = 0.0
    for question, b_rank, r_rank in rows:
        label = (question[:59] + "...") if len(question) > 59 else question
        print(f"{label:<62} {fmt(b_rank):>10} {fmt(r_rank):>10}")
        if b_rank and b_rank <= TOP_K:
            b_recall += 1
        if r_rank and r_rank <= TOP_K:
            r_recall += 1
        b_rr_sum += (1 / b_rank) if b_rank else 0
        r_rr_sum += (1 / r_rank) if r_rank else 0

    print("-" * 84)
    print(f"{'recall@' + str(TOP_K):<62} {f'{b_recall}/{n}':>10} {f'{r_recall}/{n}':>10}")
    print(f"{'mean reciprocal rank':<62} {b_rr_sum / n:>10.3f} {r_rr_sum / n:>10.3f}")


if __name__ == "__main__":
    main()
