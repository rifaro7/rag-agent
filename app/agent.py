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


def run_agent_stream(user_message: str, history: list[dict] | None, history_holder: dict):
    """Generator version of run_agent: yields text chunks as Claude produces them.

    Tool calls happen between streamed turns (the model can't stream tool
    input and text at once in a way that's useful to show live), so the
    caller sees text stream in, a pause while tools run, then more text.

    `history_holder` is filled in with the final message list once the
    generator is exhausted, since a generator can't both yield values and
    return a result to callers that only consume it via a for-loop.
    """
    messages = (history or []) + [{"role": "user", "content": user_message}]

    while True:
        with client.messages.stream(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield text
            final_message = stream.get_final_message()

        messages.append({"role": "assistant", "content": final_message.content})

        if final_message.stop_reason != "tool_use":
            history_holder["messages"] = messages
            return

        tool_results = []
        for block in final_message.content:
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
