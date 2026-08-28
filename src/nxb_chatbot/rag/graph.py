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
    grade_documents,
    guardrail,
    meal_subscription_node,
    mis_request_node,
    query_reformulator,
    retriever,
    rewrite_query,
    web_search,
    reflect_answer,
    adaptive_router,
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


def route_after_grading(state: ChatState) -> str:
    """
    After grade_documents node:
    - relevant                        → answer_generator
    - irrelevant, retries remaining   → rewrite_query (loops back to retriever)
    - irrelevant, retries exhausted   → web_search
    """
    if state.get("grade_verdict") == "relevant":
        return "answer_generator"

    if state.get("retrieval_attempts", 0) >= settings.MAX_RETRIEVAL_ATTEMPTS:
        return "web_search"

    return "rewrite_query"

def route_after_reflection(state: ChatState) -> str:
    """
    Routes after Self-RAG reflection.

    pass           -> END
    regenerate     -> answer_generator
    retrieve_again -> rewrite_query

    reflection_attempts provides the overall loop safeguard.
    """

    action = state.get("reflection_action")

    reflection_attempts = state.get("reflection_attempts", 0)
    retrieval_attempts = state.get("retrieval_attempts", 0)

    logger.info(
        f"Reflection routing → action={action}, "
        f"reflection_attempts={reflection_attempts}, "
        f"retrieval_attempts={retrieval_attempts}"
    )

    # Hard stop against infinite Self-RAG loops
    if reflection_attempts >= 3:
        logger.warning(
            "Maximum reflection attempts reached. Ending."
        )
        return "end"

    if action == "pass":
        return "end"

    if action == "regenerate":
        return "regenerate"

    if action == "retrieve_again":
        if retrieval_attempts >= settings.MAX_RETRIEVAL_ATTEMPTS:
            logger.warning(
                "Maximum retrieval attempts reached."
            )
            return "end"

        return "retrieve_again"

    logger.warning(
        f"Unknown reflection action '{action}'. Ending safely."
    )

    return "end"

def route_after_retrieval(state: ChatState) -> str:
    """
    Route retrieved documents based on adaptive query complexity.

    Simple queries skip CRAG document grading and proceed directly
    to answer generation.

    Complex queries continue through the full CRAG grading flow.
    """
    if state.get("query_route") == "simple":
        return "answer_generator"

    return "grade_documents"

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
    builder.add_node("adaptive_router", adaptive_router)
    builder.add_node("retriever", retriever)
    builder.add_node("web_search", web_search)
    builder.add_node("answer_generator", answer_generator)
    builder.add_node("reflect_answer", reflect_answer)
    builder.add_node("conversational_response", conversational_response)
    builder.add_node("mis_request", mis_request_node)
    builder.add_node("check_mis_status", check_mis_status_node)
    builder.add_node("employee_request", employee_request_node)
    builder.add_node("check_employee_request_status",check_employee_request_status_node)
    builder.add_node("grade_documents", grade_documents)
    builder.add_node("rewrite_query", rewrite_query)

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

    builder.add_edge("query_reformulator", "adaptive_router")
    builder.add_edge("adaptive_router", "retriever")
    
    builder.add_conditional_edges(
    "retriever",
    route_after_retrieval,
    {
        "answer_generator": "answer_generator",
        "grade_documents": "grade_documents",
    },
)
    
    builder.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "answer_generator": "answer_generator",
            "rewrite_query": "rewrite_query",
            "web_search": "web_search",
        },
    )
    

    builder.add_edge("rewrite_query", "retriever")
    builder.add_edge("web_search", "answer_generator")
    builder.add_edge("answer_generator","reflect_answer")

    builder.add_conditional_edges(
    "reflect_answer",
    route_after_reflection,
    {
        "end": END,
        "regenerate": "answer_generator",
        "retrieve_again": "rewrite_query",
    },
)


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