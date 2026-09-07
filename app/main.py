"""FastAPI app: serves the chat UI and /chat endpoints backed by the agent."""
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.agent import run_agent, run_agent_stream

app = FastAPI(title="RAG Agent")

# In-memory session store: {session_id: messages}. Fine for a demo/portfolio
# project; swap for Redis/a DB if this ever needs to survive a restart.
SESSIONS: dict[str, list[dict]] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    reply: str


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    history = SESSIONS.get(req.session_id, [])
    reply, updated_history = run_agent(req.message, history)
    SESSIONS[req.session_id] = updated_history
    return ChatResponse(reply=reply)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    history = SESSIONS.get(req.session_id, [])
    history_holder: dict = {}

    def token_stream():
        for chunk in run_agent_stream(req.message, history, history_holder):
            yield chunk
        SESSIONS[req.session_id] = history_holder.get("messages", history)

    return StreamingResponse(token_stream(), media_type="text/plain")


@app.get("/")
def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
