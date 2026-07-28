import logging

from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from nxb_chatbot.core.config import settings
from nxb_chatbot.rag.prompts import MEAL_CHOICE_PROMPT, MEAL_INVALID_PROMPT, rag_prompt, reformulation_prompt
from nxb_chatbot.rag.reranker import get_reranking_retriever
from nxb_chatbot.rag.schema import GuardrailResult
from nxb_chatbot.rag.services import (
    format_context,
    get_guardrail_chain,
    get_tavily_search,
    llm,
    trim_conversation,
)
from nxb_chatbot.rag.state import ChatState
from nxb_chatbot.vector_store.qdrant_client import get_vector_store

from nxb_chatbot.tools.gmail import (
    check_meal_reply,
    send_meal_acknowledgment,
    send_meal_subscription_email,
)

logger = logging.getLogger(__name__)

OFF_TOPIC_RESPONSE = (
    "I can only answer questions related to NextBridge Ltd. "
    "Please ask about company policies, procedures, or internal matters."
)



# Node 1 — Guardrail

def guardrail(state: ChatState) -> dict:
    messages = state["messages"]
    question = messages[-1].content

    logger.info(f"Running guardrail for: {question}")

    chain = get_guardrail_chain()
    result: GuardrailResult = chain.invoke({"question": question})

    logger.info(
        f"Guardrail: passed={result.passed} | intent={result.intent} | reason={result.reason}"
    )

    if not result.passed:
        return {
            "guardrail_passed": False,
            "meal_intent": None,
            "messages": [AIMessage(content=OFF_TOPIC_RESPONSE)],
        }

    return {
        "guardrail_passed": True,
        "meal_intent": result.intent if result.intent != "general_query" else None,
    }


def _parse_meal_preference(user_input: str) -> str | None:
    """
    Uses the LLM to extract meal preference from any natural language input.
    Falls back to keyword matching in case LLM echoes extra words.
    Returns one of: Lunch | Dinner | Both | Roti Only — or None if unclear.
    """
    prompt = (
        f"The user was asked to choose a meal subscription type from these options:\n"
        f"1. Lunch\n2. Dinner\n3. Both (Lunch + Dinner)\n4. Roti Only\n\n"
        f"The user replied: \"{user_input}\"\n\n"
        f"Return ONLY one of these exact strings with no extra words:\n"
        f"Lunch\nDinner\nBoth\nRoti Only\n\n"
        f"If you cannot determine their choice, return ONLY: UNCLEAR"
    )

    response = llm.invoke(prompt)
    parsed = response.content.strip()

    # Exact match first
    if parsed in ("Lunch", "Dinner", "Both", "Roti Only"):
        return parsed

    # Fuzzy keyword fallback — handles "Dinner only", "Both (Lunch + Dinner)" etc.
    parsed_lower = parsed.lower()
    if "roti" in parsed_lower:
        return "Roti Only"
    if "both" in parsed_lower or ("lunch" in parsed_lower and "dinner" in parsed_lower):
        return "Both"
    if "lunch" in parsed_lower:
        return "Lunch"
    if "dinner" in parsed_lower:
        return "Dinner"

    return None

# Node 2 — Query Reformulator

def query_reformulator(state: ChatState) -> dict:
    """
    First turn  → return original question as standalone_query.
    Follow-up   → reformulate using chat history.
    """
    messages = state["messages"]
    current_question = messages[-1].content

    if len(messages) == 1:
        logger.info("First turn — skipping reformulation.")
        return {"standalone_query": current_question}

    logger.info("Follow-up turn — reformulating query.")

    chain = reformulation_prompt | llm
    response = chain.invoke(
        {
            "messages": messages[:-1],
            "question": current_question,
        }
    )

    standalone_query = response.content.strip()
    logger.info(f"Reformulated query: {standalone_query}")

    return {"standalone_query": standalone_query}


# Node 3 — Retriever
def retriever(state: ChatState) -> dict:
    """
    Hybrid search against Qdrant using standalone_query.
    Applies metadata filters if set in state.
    Reranks results and checks score threshold.
    Sets web_search_used = True if no relevant chunks found.
    """
    query = state["standalone_query"]
    filters = state.get("retrieval_filters")

    logger.info(f"Retrieving docs for query: {query}")

    vector_store = get_vector_store()
    search_kwargs = {"k": settings.RETRIEVER_TOP_K}

    if filters:
        search_kwargs["filter"] = filters
        logger.info(f"Applying retrieval filters: {filters}")

    base_retriever = vector_store.as_retriever(
        search_kwargs=search_kwargs,
    )

    reranking_retriever = get_reranking_retriever(base_retriever)
    docs = reranking_retriever.invoke(query)

    # Convert to serializable dicts + fix numpy types
    serializable_docs = [
        {
            "page_content": doc.page_content,
            "metadata": {
                k: float(v) if hasattr(v, "item") else v
                for k, v in doc.metadata.items()
            },
        }
        for doc in docs
    ]

    # Check if any chunk meets the relevance threshold
    max_score = max(
        (d["metadata"].get("relevance_score", 0.0) for d in serializable_docs),
        default=0.0,
    )

    web_search_needed = max_score < settings.RERANK_SCORE_THRESHOLD

    if web_search_needed:
        logger.info(
            f"Max rerank score {max_score:.4f} below threshold "
            f"{settings.RERANK_SCORE_THRESHOLD} — will trigger web search."
        )
    else:
        logger.info(
            f"Retrieved and reranked → {len(serializable_docs)} chunks. "
            f"Max score: {max_score:.4f}"
        )

    return {
        "retrieved_docs": serializable_docs,
        "web_search_used": web_search_needed,
    }


# Node 4 — Web Search

def web_search(state: ChatState) -> dict:
    """
    Fallback when RAG retrieval score is below threshold.
    Scopes search to NextBridge by injecting company name into query.
    Results formatted as retrieved_docs for answer_generator.
    """
    query = state["standalone_query"]
    scoped_query = f"NextBridge {query}"

    logger.info(f"Triggering web search for: {scoped_query}")

    tavily = get_tavily_search()
    results = tavily.invoke(scoped_query)

    web_docs = [
        {
            "page_content": r.get("content", ""),
            "metadata": {
                "source": r.get("url", "web"),
                "file_name": "web_search",
                "page": 0,
                "has_table": False,
                "relevance_score": r.get("score", 0.0),
                "web_result": True,
            },
        }
        for r in results
    ]

    logger.info(f"Web search returned {len(web_docs)} results.")

    return {"retrieved_docs": web_docs}

# Node 5 — Answer Generator

def answer_generator(state: ChatState) -> dict:
    """
    Generates final answer using retrieved context + trimmed chat history.
    """
    context = format_context(state)
    trimmed_messages = trim_conversation(state)

    logger.info("Generating answer.")

    chain = rag_prompt | llm
    response = chain.invoke(
        {
            "context": context,
            "messages": trimmed_messages,
        }
    )

    logger.info("Answer generated.")

    return {"messages": [response]}


def _get_latest_human_message(messages: list) -> str:
    """Returns the content of the most recent human message."""
    for msg in reversed(messages):
        if msg.__class__.__name__ == "HumanMessage":
            return msg.content
    return ""

# ---------------------------------------------------------------------------
# Node 6 — Meal subscription
# ---------------------------------------------------------------------------

def meal_subscription_node(state: ChatState) -> dict:
    """
    meal_data.step tracks where we are in the flow.
    route_entry bypasses guardrail so mid-flow messages reach this node directly.
    """
    meal = state.get("meal_data") or {}
    latest = _get_latest_human_message(state["messages"])

    if meal.get("email_sent"):
        return {
            "messages": [AIMessage(
                content=(
                    f"Your **{meal.get('preference', 'meal')}** subscription is already submitted. "
                    f"Ask *'What is the status of my meal subscription?'* to check for updates."
                )
            )]
        }

    step = meal.get("step", "start")

    if step == "start":
        return {
            "meal_data": {**meal, "step": "waiting_preference", "in_progress": True},
            "messages": [AIMessage(content=MEAL_CHOICE_PROMPT)],
        }

    # ── Step 2: Parse meal preference — ask for name ────────────────────────
    if step == "waiting_preference":
        preference = _parse_meal_preference(latest)
        if not preference:
            return {
                "meal_data": {**meal},
                "messages": [AIMessage(content=MEAL_INVALID_PROMPT)],
            }
        return {
            "meal_data": {**meal, "step": "waiting_name", "preference": preference},
            "messages": [AIMessage(
                content="Please enter your **full name** as it appears in HR records:"
            )],
        }

    # ── Step 3: Save name — ask for employee ID ─────────────────────────────
    if step == "waiting_name":
        return {
            "meal_data": {**meal, "step": "waiting_emp_id", "name": latest.strip()},
            "messages": [AIMessage(
                content="Please enter your **Employee ID** (e.g. NXB-0042):"
            )],
        }

    # ── Step 4: Save emp ID — send email ────────────────────────────────────
    if step == "waiting_emp_id":
        emp_id = latest.strip()
        preference = meal.get("preference", "")
        name = meal.get("name", "")

        logger.info(f"Sending meal subscription: {name}, {emp_id}, {preference}")

        result = send_meal_subscription_email.invoke({
            "name": name,
            "employee_id": emp_id,
            "preference": preference,
        })

        thread_id: str | None = None
        if "thread_id=" in result:
            thread_id = result.split("thread_id=")[-1].strip() or None

        return {
            "meal_data": {
                **meal,
                "step": "completed",
                "employee_id": emp_id,
                "email_sent": True,
                "thread_id": thread_id,
                "in_progress": False,
            },
            "messages": [AIMessage(
                content=(
                    f"Done, **{name}**! Your **{preference}** subscription request "
                    f"(ID: {emp_id}) has been sent to the meals department.\n\n"
                    f"Ask *'What is the status of my meal subscription?'* anytime to check for a reply."
                )
            )],
        }

    return {
        "messages": [AIMessage(content="Something went wrong. Please say 'I want to subscribe to meals' to start again.")]
    }

# Node 7 — Check meal subscription status

def check_meal_status_node(state: ChatState) -> dict:
    """
    Checks for a department reply.
    If reply found: shows it and asks for ack confirmation.
    If waiting_for_ack: processes yes/no from the latest message.
    """
    meal = state.get("meal_data") or {}
    latest = _get_latest_human_message(state["messages"])

    preference = meal.get("preference")
    name       = meal.get("name", "the employee")
    emp_id     = meal.get("employee_id", "N/A")
    thread_id  = meal.get("thread_id")

    if not meal.get("email_sent") or not preference:
        return {
            "messages": [AIMessage(
                content=(
                    "I don't have a submitted subscription for this session. "
                    "Say *'I want to subscribe to meals'* to start one."
                )
            )]
        }

    if meal.get("acknowledged"):
        return {
            "messages": [AIMessage(
                content=f"Your **{preference}** subscription was already acknowledged. You're all set!"
            )]
        }

    # ── Waiting for yes/no on acknowledgment ────────────────────────────────
    if meal.get("waiting_for_ack"):
        if latest.strip().lower() in ("yes", "y"):
            send_meal_acknowledgment.invoke({
                "name": name,
                "employee_id": emp_id,
            })
            return {
                "meal_data": {**meal, "acknowledged": True, "waiting_for_ack": False},
                "messages": [AIMessage(
                    content="✅ Acknowledgment sent to the meals department. You're all set!"
                )],
            }
        return {
            "meal_data": {**meal, "waiting_for_ack": False},
            "messages": [AIMessage(
                content="Okay, acknowledgment skipped. Ask for the status again anytime to send it."
            )],
        }

    # ── Check for reply via @tool ────────────────────────────────────────────
    logger.info(f"Checking meal reply for thread_id={thread_id}")
    reply_body = check_meal_reply.invoke({"thread_id": thread_id or ""})

    if reply_body == "NO_REPLY":
        return {
            "messages": [AIMessage(
                content=f"No reply yet from the meals department for your **{preference}** subscription. Please check back later."
            )]
        }

    # ── Reply found — show it and ask for ack ───────────────────────────────
    ack_draft = (
        f"Dear Meals Coordinator,\n\n"
        f"Thank you for your response regarding the meal subscription for "
        f"{name} (ID: {emp_id}).\n\n"
        f"We acknowledge receipt and will act accordingly.\n\n"
        f"Regards,\nNXB Chatbot System"
    )

    return {
        "meal_data": {**meal, "waiting_for_ack": True},
        "messages": [AIMessage(
            content=(
                f"📬 The meals department has replied!\n\n"
                f"**Their reply:** _{reply_body[:500]}_\n\n"
                f"---\n"
                f"**Draft acknowledgment:**\n```\n{ack_draft}\n```\n\n"
                f"Should I send this? Reply **yes** to send or **no** to skip."
            )
        )],
    }