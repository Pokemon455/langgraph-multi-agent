from langchain_core.messages import SystemMessage, ToolMessage
from state import State
from llms import answer_LLM

_SYSTEM_PROMPT = """You are a helpful AI assistant. Rules:

LANGUAGE:
- Roman Urdu query → reply in Roman Urdu
- English query → reply in English
- Never use Urdu/Hindi script (no ا ب پ)

FORMAT:
- Plain text only — no markdown, no bullet points, no bold, no headers
- Short and natural — 2 to 4 lines max
- No filler phrases like "Sure!", "Great!", "Of course!"

TOOL RESULT:
- If a tool result is provided, explain it clearly in 1-2 lines
- State key values directly (version number, sum, price, etc.)
- Do not repeat the raw tool output verbatim"""


async def answer_node(state: State):
    msgs = state["messages"]

    tool_out = next(
        (m.content for m in reversed(msgs) if isinstance(m, ToolMessage)),
        None
    )

    # MCP tool output comes as list[dict] — parse to plain string
    if isinstance(tool_out, list):
        tool_out = "\n".join(
            item.get("text", "") for item in tool_out if item.get("type") == "text"
        ).strip()

    sp = _SYSTEM_PROMPT
    if tool_out:
        sp += f"\n\nTool result:\n{tool_out}"

    history = [m for m in msgs[-10:] if not isinstance(m, ToolMessage)]
    resp = await answer_LLM.ainvoke([SystemMessage(content=sp)] + history)
    return {"messages": [resp]}
