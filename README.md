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
ChromaDB vector store (app/rag.py) ── populated by app/ingest.py
```

Each turn, the app sends the conversation to Claude along with a list of
tool definitions. Claude either replies directly, or asks to call a tool
(e.g. `search_documents`). The backend runs that tool in Python, feeds the
result back to Claude as a new message, and repeats until Claude has enough
information to give a final answer.

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
- Add an eval set (sample questions + expected answers) to measure
  retrieval quality
- Swap ChromaDB for a managed vector DB (Pinecone/pgvector) for larger,
  persistent corpora
