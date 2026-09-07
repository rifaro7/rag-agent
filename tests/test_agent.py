"""Tests the agent loop's own logic — tool selection, execution, and
response handling — without calling the real (paid) Claude API.

Only the LLM call itself is mocked. Retrieval runs against a real,
throwaway Chroma collection, and the tool dispatch in app/tools.py runs
for real, so this actually exercises the code that matters: does the loop
correctly recognize a tool_use response, run the right tool with the right
arguments, feed the real result back, and terminate on a text response.
"""
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-placeholder")

from unittest.mock import MagicMock, patch

import chromadb

from app import agent, rag


def _tool_use_response(name, tool_input, call_id="toolu_test"):
    # `name` can't be passed to MagicMock()'s constructor — that kwarg sets
    # the mock's own repr identity, not a `.name` attribute — so it's set
    # afterward as a plain attribute assignment instead.
    block = MagicMock(type="tool_use", input=tool_input, id=call_id)
    block.name = name
    return MagicMock(stop_reason="tool_use", content=[block])


def _text_response(text):
    block = MagicMock(type="text", text=text)
    return MagicMock(stop_reason="end_turn", content=[block])


def test_agent_loop_executes_tool_and_returns_grounded_answer(tmp_path, monkeypatch):
    # Point the vector store at a throwaway location and seed one real fact,
    # so retrieval has something genuine to find.
    monkeypatch.setattr(rag, "_client", chromadb.PersistentClient(path=str(tmp_path)))
    rag.get_collection().upsert(
        ids=["doc-0"],
        documents=["Full-time employees accrue 1.5 days of paid time off per month."],
        metadatas=[{"source": "leave_policy.txt"}],
    )

    # Simulated Claude turn 1: decide to search the documents.
    turn1 = _tool_use_response("search_documents", {"query": "PTO accrual per month"})
    # Simulated Claude turn 2: answer using whatever the tool actually returned.
    turn2 = _text_response("Employees earn 1.5 days of PTO per month.")

    with patch.object(agent.client.messages, "create", side_effect=[turn1, turn2]) as mock_create:
        final_text, messages = agent.run_agent("How much PTO do employees earn per month?")

    assert mock_create.call_count == 2, "expected exactly one tool round-trip"
    assert "1.5 days" in final_text

    tool_result_text = messages[2]["content"][0]["content"]
    assert "leave_policy.txt" in tool_result_text, "tool result should be the real retrieved chunk"


def test_agent_loop_returns_directly_when_no_tool_needed(monkeypatch):
    direct_answer = _text_response("2 + 2 is 4.")

    with patch.object(agent.client.messages, "create", return_value=direct_answer) as mock_create:
        final_text, messages = agent.run_agent("What's 2 + 2?")

    assert mock_create.call_count == 1, "should not call a tool for something Claude can answer directly"
    assert final_text == "2 + 2 is 4."
