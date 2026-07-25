import logging

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from psycopg_pool import AsyncConnectionPool

from nxb_chatbot.core.config import settings
from nxb_chatbot.rag.nodes import answer_generator, query_reformulator, retriever
from nxb_chatbot.rag.state import ChatState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------

def _build_graph() -> StateGraph:
    """
    Defines nodes and edges of the RAG graph.
    Returns uncompiled graph — checkpointer is attached at runtime.
    """
    builder = StateGraph(ChatState)

    # Nodes
    builder.add_node("query_reformulator", query_reformulator)
    builder.add_node("retriever", retriever)
    builder.add_node("answer_generator", answer_generator)

    # Edges
    builder.add_edge(START, "query_reformulator")
    builder.add_edge("query_reformulator", "retriever")
    builder.add_edge("retriever", "answer_generator")
    builder.add_edge("answer_generator", END)

    return builder


# ---------------------------------------------------------------------------
# Compiled Graph Factory
# ---------------------------------------------------------------------------

async def get_compiled_graph():
    """
    Creates an async connection pool, sets up AsyncPostgresSaver,
    runs migrations, and returns the compiled graph.

    This should be called once during FastAPI lifespan startup
    and the result cached for reuse.
    """
    connection_pool = AsyncConnectionPool(
        conninfo=settings.CHECKPOINTER_DATABASE_URL,
        max_size=10,
        open=False,
        kwargs={"autocommit": True},
    )

    await connection_pool.open()
    logger.info("Postgres connection pool opened.")

    checkpointer = AsyncPostgresSaver(connection_pool)

    # Creates checkpointer tables in Postgres if they don't exist
    await checkpointer.setup()
    logger.info("AsyncPostgresSaver tables ready.")

    graph = _build_graph().compile(checkpointer=checkpointer)
    logger.info("RAG graph compiled successfully.")

    return graph, connection_pool