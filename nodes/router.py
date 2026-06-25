"""Router node — classifies query and decides which tool to use."""
from langchain_core.messages import SystemMessage, HumanMessage
from state import State, RouterDecision
from llms import router_LLM

_PROMPT = """Classify the query into one of:
- python_tool : code, pip, install, version, "karo", "chalao"
- rag         : document, file, PDF, uploaded, "mera data"
- web_search  : latest, news, today, current, price, weather
- direct      : greetings, general knowledge, simple questions

When in doubt between python_tool and direct → python_tool."""


async def router_node(state: State):
    try:
        r = await router_LLM.with_structured_output(RouterDecision).ainvoke([
            SystemMessage(content=_PROMPT + f"\nfile_uploaded={state.get('file_uploaded', False)}"),
            HumanMessage(content=state["messages"][-1].content)
        ])
        return {"router_decision": r.decision, "reasoning": r.reasoning}
    except Exception as e:
        return {"router_decision": "direct", "reasoning": str(e)}
