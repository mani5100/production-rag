import logging

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from psycopg_pool import AsyncConnectionPool

from nxb_chatbot.core.config import settings
from nxb_chatbot.rag.nodes import (
    answer_generator,
    check_employee_request_status_node,
    check_meal_status_node,
    check_mis_status_node,
    conversational_response,
    employee_request_node,
    guardrail,
    meal_subscription_node,
    mis_request_node,
    query_reformulator,
    retriever,
    web_search,
)
from nxb_chatbot.rag.state import ChatState

logger = logging.getLogger(__name__)

def route_entry(state: ChatState) -> str:
    meal = state.get("meal_data") or {}
    mis = state.get("mis_data") or {}
    employee_request = state.get("employee_request_data") or {}

    if meal.get("in_progress") and not meal.get("email_sent"):
        return "meal_subscription"

    if meal.get("waiting_for_ack") and not meal.get("acknowledged"):
        return "check_meal_status"

    if mis.get("in_progress") and not mis.get("email_sent"):
        return "mis_request"

    if mis.get("waiting_for_ack") and not mis.get("acknowledged"):
        return "check_mis_status"

    if (
        employee_request.get("in_progress")
        and not employee_request.get("email_sent")
    ):
        return "employee_request"

    if (
        employee_request.get("waiting_for_ack")
        and not employee_request.get("acknowledged")
    ):
        return "check_employee_request_status"

    return "guardrail"


def route_after_guardrail(state: ChatState) -> str:
    if not state.get("guardrail_passed"):
        return END

    intent = state.get("route_intent")

    if intent == "meal_subscription":
        return "meal_subscription"

    if intent == "meal_status_check":
        return "check_meal_status"

    if intent == "mis_request":
        return "mis_request"

    if intent == "mis_status_check":
        return "check_mis_status"

    if intent == "conversational":
        return "conversational_response"
    
    if intent == "employee_request":
        return "employee_request"

    if intent == "employee_request_status":
        return "check_employee_request_status"

    return "query_reformulator"


def route_after_retriever(state: ChatState) -> str:
    """
    After retriever node:
    - web_search_used = True  → fallback to web_search
    - web_search_used = False → go to answer_generator
    """
    if state.get("web_search_used"):
        return "web_search"
    return "answer_generator"

# Graph Builder
def _build_graph() -> StateGraph:
    """
    Defines nodes and edges of the RAG graph.
    Returns uncompiled graph — checkpointer attached at runtime.
    """
    builder = StateGraph(ChatState)

    # Nodes
    builder.add_node("guardrail", guardrail)
    builder.add_node("query_reformulator", query_reformulator)
    builder.add_node("retriever", retriever)
    builder.add_node("web_search", web_search)
    builder.add_node("answer_generator", answer_generator)
    builder.add_node("conversational_response", conversational_response)
    builder.add_node("mis_request", mis_request_node)
    builder.add_node("check_mis_status", check_mis_status_node)
    builder.add_node("employee_request", employee_request_node)
    builder.add_node("check_employee_request_status",check_employee_request_status_node)

    builder.add_node("meal_subscription", meal_subscription_node)
    builder.add_node("check_meal_status", check_meal_status_node)
    
    
    # Entry point
    builder.add_conditional_edges(
        START,
        route_entry,
        {
            "guardrail": "guardrail",
            "meal_subscription": "meal_subscription",
            "check_meal_status": "check_meal_status",
            "mis_request": "mis_request",
            "check_mis_status": "check_mis_status",
            "employee_request": "employee_request",
            "check_employee_request_status": (
                "check_employee_request_status"
            ),
        },
    )
    
    builder.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {
            "meal_subscription": "meal_subscription",
            "check_meal_status": "check_meal_status",
            "mis_request": "mis_request",
            "check_mis_status": "check_mis_status",
            "employee_request": "employee_request",
            "check_employee_request_status": (
                "check_employee_request_status"
            ),
            "conversational_response": "conversational_response",
            "query_reformulator": "query_reformulator",
            END: END,
        },
    )

    builder.add_edge("query_reformulator", "retriever")
    builder.add_conditional_edges(
        "retriever",
        route_after_retriever,
        {
            "web_search": "web_search",
            "answer_generator": "answer_generator",
        },
    )

    builder.add_edge("web_search", "answer_generator")

    builder.add_edge("answer_generator", END)
    builder.add_edge("meal_subscription", END)
    builder.add_edge("check_meal_status", END)
    builder.add_edge("mis_request", END)
    builder.add_edge("check_mis_status", END)
    builder.add_edge("conversational_response", END)
    builder.add_edge("employee_request", END)
    builder.add_edge("check_employee_request_status", END)

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