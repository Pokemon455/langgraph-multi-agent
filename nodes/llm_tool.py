from langchain_core.messages import SystemMessage, ToolMessage
from state import State
from llms import answer_LLM

_TOOL_MAP = {
    "python_tool": "run_python",
    "rag":         "RAG",
    "web_search":  "search_web",
}


async def llm_tool_node(state: State, all_tools: list):
    msgs    = state["messages"]
    dec     = state.get("router_decision")
    itr     = state.get("iteration_count", 0)

    if isinstance(msgs[-1], ToolMessage):
        return {"messages": [], "iteration_count": itr}

    forced = _TOOL_MAP.get(dec)
    sp = f"""You are an AI assistant. Routing decision: {dec}.
You MUST call the tool: {forced}
- run_python  → write complete Python code using print() for output
- RAG         → pass the user question exactly as-is
- search_web  → pass the user question exactly as-is
Do NOT reply with text. Only make a tool call."""

    resp = await answer_LLM.bind_tools(all_tools, tool_choice=forced).ainvoke(
        [SystemMessage(content=sp)] + msgs[-6:]
    )
    return {"messages": [resp], "iteration_count": itr + 1}
