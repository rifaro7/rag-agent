"""Agentic loop: sends messages to Claude, executes any tool calls it makes,
and feeds results back until Claude produces a final text answer."""
import os

from anthropic import Anthropic
from dotenv import load_dotenv

from app.tools import TOOL_SCHEMAS, run_tool

load_dotenv()

MODEL = "claude-sonnet-4-5"
SYSTEM_PROMPT = (
    "You are a helpful research assistant. You have tools to search the user's "
    "own documents, search the web, and do arithmetic. Prefer search_documents "
    "for questions about the user's files; use web_search for general/current "
    "info. Cite which source (document name or URL) an answer came from when relevant."
)

_api_key = os.environ.get("ANTHROPIC_API_KEY")
if not _api_key:
    raise RuntimeError(
        "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key "
        "(get one at https://console.anthropic.com/settings/keys)."
    )
client = Anthropic(api_key=_api_key)


def run_agent(user_message: str, history: list[dict] | None = None) -> tuple[str, list[dict]]:
    """Runs one turn of the agent loop. Returns (final_text, updated_history)."""
    messages = (history or []) + [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            final_text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            messages.append({"role": "assistant", "content": response.content})
            return final_text, messages

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = run_tool(block.name, block.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )
        messages.append({"role": "user", "content": tool_results})
