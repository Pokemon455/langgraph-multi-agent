"""Answer node — generates final response using tool output + chat history."""
from langchain_core.messages import SystemMessage, ToolMessage
from state import State
from llms import answer_LLM

_PROMPT = """You are a helpful AI assistant.
- Roman Urdu query → reply in Roman Urdu | English query → reply in English
- Plain text only — no markdown, bullets, or headers
- 2-4 lines max, no filler phrases like "Sure!" or "Great!"
- If tool result provided → explain it clearly in 1-2 lines"""


def _get_tool_output(msgs):
    for m in reversed(msgs):
        if isinstance(m, ToolMessage):
            out = m.content
            if isinstance(out, list):
                return "\n".join(i.get("text", "") for i in out if i.get("type") == "text").strip()
            return out
    return None


async def answer_node(state: State):
    msgs     = state["messages"]
    tool_out = _get_tool_output(msgs)
    prompt   = _PROMPT + (f"\n\nTool result:\n{tool_out}" if tool_out else "")
    history  = [m for m in msgs[-10:] if not isinstance(m, ToolMessage)]
    return {"messages": [await answer_LLM.ainvoke([SystemMessage(content=prompt)] + history)]}
