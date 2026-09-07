# RAG Agent

A tool-calling AI agent that answers questions by retrieving from your own
documents (RAG), searching the web, or doing math — built with Claude,
ChromaDB, and FastAPI.

## Architecture

```
Browser (static/index.html)
        │  POST /chat
        ▼
FastAPI (app/main.py)
        │
        ▼
Agent loop (app/agent.py)  ──calls──▶  Claude (tool-calling)
        │                                     │
        │  runs whichever tool Claude picks   │
        ▼                                     ▼
app/tools.py:  search_documents (RAG) · web_search · calculator
        │
        ▼
ChromaDB vector search (top 15) ── populated by app/ingest.py
        │
        ▼
cross-encoder re-ranking (top 4) ── app/rag.py
```

Each turn, the app sends the conversation to Claude along with a list of
tool definitions. Claude either replies directly, or asks to call a tool
(e.g. `search_documents`). The backend runs that tool in Python, feeds the
result back to Claude as a new message, and repeats until Claude has enough
information to give a final answer.

Retrieval is two-stage rather than a single vector lookup: Chroma's cosine
similarity pulls a wide pool of 15 candidates cheaply, then a cross-encoder
(`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-scores each candidate against the
actual question and keeps the top 4. Vector similarity compares two
independent embeddings; a cross-encoder reads the question and the candidate
passage together, which is slower per-comparison but generally more precise
— the standard fix for cases where topically-similar-but-wrong passages
outrank the one that actually answers the question.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit .env and add your ANTHROPIC_API_KEY
```

Get a key at https://console.anthropic.com/settings/keys.

## Ingest documents (optional but recommended)

Drop some PDFs or text files into `documents/`, then run:

```bash
python -m app.ingest
```

## Run

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000

## Deploy

This repo includes a `Dockerfile` and `render.yaml`:

1. Push this repo to GitHub.
2. On [Render](https://dashboard.render.com), click **New > Blueprint** and
   point it at the repo — it reads `render.yaml` automatically and sets up
   a free Docker web service.
3. When prompted, paste `ANTHROPIC_API_KEY` as the secret env var (it's
   marked `sync: false` in `render.yaml` so it's never committed).
4. Render builds the Dockerfile and gives you a public URL.

The same `Dockerfile` works on Fly.io or Railway. Note: on a free tier the
container's disk isn't guaranteed to persist across restarts, so
`docker-entrypoint.sh` re-runs ingestion on every startup — fine for a
small document set, but swap in a managed vector DB (e.g. Pinecone) or a
persistent volume for a larger corpus.

## Retrieval quality (measured, not assumed)

`eval/` holds a small, self-contained evaluation: 6 synthetic documents
(including deliberate near-duplicate "distractor" pairs — two API docs, two
leave policies, two onboarding checklists that share vocabulary but differ
in specifics) and 13 questions, some paraphrased to avoid reusing the
source's exact wording. It runs against its own in-memory Chroma collection,
never touching your real `chroma_db/`.

```bash
python -m eval.run_eval
```

Actual output on this corpus:

```
recall@4                                                            13/13      13/13
mean reciprocal rank                                                1.000      1.000
```

Both plain vector search and the cross-encoder re-ranker land every question
in first place. That's an honest result, not the one I expected to write
down — and it's informative rather than a failure: at 6 documents, there
just isn't enough ambiguity in the embedding space for re-ranking to have
anything to correct. The value of an eval harness isn't only "prove the
optimization helped" — it's catching exactly this, that a change everyone
assumes is an improvement may not move the needle at a given scale, before
you ship it based on intuition. Re-ranking is worth keeping anyway because
its advantage shows up as the candidate pool gets larger or noisier than a
6-document demo corpus; the harness is what would tell you if or when that
stops being true for your real document set.

## Design notes

- **RAG over stuffing the prompt**: keeps token usage and cost bounded as
  the document set grows, and only pulls in passages relevant to the
  current question.
- **Local embeddings (sentence-transformers)**: no extra API cost or
  network dependency for the ingestion step.
- **In-memory session store**: fine for a single-instance demo; swap for
  Redis/Postgres for anything that needs to survive a restart or scale
  past one process.

## Roadmap

- Swap the in-memory session store for Redis/Postgres
- Grow `eval/` until it actually discriminates between retrieval strategies
  (larger corpus, harder distractors) rather than saturating
- Swap ChromaDB for a managed vector DB (Pinecone/pgvector) for larger,
  persistent corpora
