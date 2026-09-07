"""Tool definitions and implementations for the agent's tool-calling loop."""
import ast
import operator

from app.rag import query as rag_query

TOOL_SCHEMAS = [
    {
        "name": "search_documents",
        "description": (
            "Search the user's ingested documents (PDFs/notes) for relevant passages. "
            "Use this whenever the question could be answered from the user's own files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for."}
            },
            "required": ["query"],
        },
    },
    {
        "name": "web_search",
        "description": "Search the public web for current or general-knowledge information.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."}
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculator",
        "description": "Evaluate a numeric arithmetic expression, e.g. '12 * (3 + 4) / 2'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "A math expression."}
            },
            "required": ["expression"],
        },
    },
]

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression")


def calculator(expression: str) -> str:
    try:
        result = _safe_eval(ast.parse(expression, mode="eval").body)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


def search_documents(query: str) -> str:
    hits = rag_query(query)
    if not hits:
        return "No matching documents found (have you run `python -m app.ingest`?)."
    return "\n\n".join(f"[{h['source']}]: {h['text']}" for h in hits)


def web_search(query: str) -> str:
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No web results found."
        return "\n\n".join(f"{r['title']}: {r['body']} ({r['href']})" for r in results)
    except Exception as e:
        return f"Web search failed: {e}"


DISPATCH = {
    "search_documents": lambda inp: search_documents(inp["query"]),
    "web_search": lambda inp: web_search(inp["query"]),
    "calculator": lambda inp: calculator(inp["expression"]),
}


def run_tool(name: str, tool_input: dict) -> str:
    if name not in DISPATCH:
        return f"Unknown tool: {name}"
    return DISPATCH[name](tool_input)
