import asyncio
import nest_asyncio
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from graph import build_graph
import config

nest_asyncio.apply()
config.validate()


async def run(query: str, thread_id: str = "1") -> str:
    """
    Run the LangGraph multi-agent chatbot.

    Args:
        query:     User input string.
        thread_id: Conversation ID — same ID = same memory thread.

    Returns:
        Agent response as a plain string.
    """
    g, _ = await build_graph()

    async with AsyncPostgresSaver.from_conn_string(config.DATABASE_URL) as cp:
        await cp.setup()
        bot = g.compile(checkpointer=cp)
        res = await bot.ainvoke(
            {
                "messages":       [HumanMessage(content=query)],
                "file_uploaded":  False,
                "iteration_count": 0
            },
            config={"configurable": {"thread_id": thread_id}}
        )
        return res["messages"][-1].content


if __name__ == "__main__":
    result = asyncio.get_event_loop().run_until_complete(
        run("python version check karo", thread_id="1")
    )
    print(result)
