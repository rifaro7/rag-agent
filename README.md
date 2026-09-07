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

**How the agent loop works** (the core AI-engineering concept here): each
turn, the app sends the conversation to Claude along with a list of tool
definitions. Claude either replies directly, or asks to call a tool (e.g.
`search_documents`). The backend runs that tool in Python, feeds the result
back to Claude as a new message, and repeats until Claude has enough
information to give a final answer. This is the same pattern used by
production AI agents and coding assistants.

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

## What to say about this on your resume/GitHub

- "Built an agentic RAG assistant using Claude's tool-calling API, ChromaDB
  for vector search, and FastAPI, with a document ingestion pipeline
  (chunking + local embeddings via sentence-transformers)."
- Talking points for interviews: why RAG instead of stuffing everything into
  the prompt (context limits, cost, relevance), how the tool-calling loop
  works, chunking/overlap tradeoffs, and how you'd swap ChromaDB for a
  production vector DB (Pinecone/pgvector) at scale.

## Next steps to extend it (good follow-up portfolio work)

- Add streaming responses (Claude's streaming API + Server-Sent Events)
- Swap the in-memory session store for Redis/Postgres so it survives restarts
- Add a real eval set (sample questions + expected answers) to measure
  retrieval quality
- Deploy it (Render/Fly.io/Railway) and put a live demo link in your resume
