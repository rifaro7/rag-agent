Drop `.pdf`, `.txt`, or `.md` files here, then run:

```
python -m app.ingest
```

This chunks and embeds them into the local Chroma vector store so the agent's
`search_documents` tool can retrieve them.
