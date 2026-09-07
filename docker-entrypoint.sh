#!/bin/sh
set -e

# Re-embed whatever's in documents/ on each start, since the vector store
# isn't guaranteed to persist across deploys on a free-tier host.
python -m app.ingest

exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
