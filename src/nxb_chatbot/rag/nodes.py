from datetime import date
import json
import logging

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from nxb_chatbot.rag.semantic_cache import check_semantic_cache, store_semantic_cache

from nxb_chatbot.core.config import settings
from nxb_chatbot.rag.prompts import (
    MEAL_CHOICE_PROMPT,
    MEAL_INVALID_PROMPT,
    acknowledgement_confirmation_prompt,
    conversational_prompt,
    employee_confirmation_prompt,
    employee_request_prompt,
    gm_acknowledgement_prompt,
    rag_prompt,
    reformulation_prompt,
    rewrite_prompt,
    reflection_prompt,
    adaptive_router_prompt,
)
from nxb_chatbot.rag.reranker import get_multi_query_retriever, get_reranking_retriever
from nxb_chatbot.rag.schema import (
    AcknowledgementConfirmationDecision,
    AnswerReflection,
    EmployeeConfirmationDecision,
    EmployeeRequestDecision,
    GMAcknowledgementResult,
    GuardrailResult,
    QueryReformulation,
    QueryRoute,
)
from nxb_chatbot.rag.services import (
    _employee_request_view,
    _extract_tracking_data,
    _merge_non_null_values,
    format_context,
    get_grading_chain,
    get_guardrail_chain,
    get_tavily_search,
    llm,
    trim_conversation,
)

from nxb_chatbot.rag.state import ChatState
from nxb_chatbot.vector_store.qdrant_client import get_vector_store

from nxb_chatbot.tools.gmail import (
    check_gm_employee_request_reply,
    check_meal_reply,
    check_mis_reply,
    send_employee_request_to_gm,
    send_gm_employee_request_acknowledgement,
    send_meal_acknowledgment,
    send_meal_subscription_email,
    send_mis_acknowledgment,
    send_mis_request_email,
)

logger = logging.getLogger(__name__)

OFF_TOPIC_RESPONSE = (
    "I can only answer questions related to NextBridge Ltd. "
    "Please ask about company policies, procedures, or internal matters."
)


# Node 1 — Guardrail


def guardrail(state: ChatState, config: RunnableConfig) -> dict:
    messages = state["messages"]
    question = messages[-1].content

    logger.info(f"Running guardrail for: {question}")

    chain = get_guardrail_chain()
    result: GuardrailResult = chain.invoke({"question": question}, config=config)

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
        "route_intent": result.intent,
    }


def conversational_response(state: ChatState, config: RunnableConfig) -> dict:
    messages = trim_conversation(state)

    response = (conversational_prompt | llm).invoke(
        {"messages": messages}, config=config
    )

    return {
        "messages": [response],
        "retrieved_docs": [],
        "web_search_used": False,
    }


def _parse_meal_preference(
    user_input: str, config: RunnableConfig | None = None
) -> str | None:
    """
    Uses the LLM to extract meal preference from any natural language input.
    Falls back to keyword matching in case LLM echoes extra words.
    Returns one of: Lunch | Dinner | Both | Roti Only — or None if unclear.
    """
    prompt = (
        f"The user was asked to choose a meal subscription type from these options:\n"
        f"1. Lunch\n2. Dinner\n3. Both (Lunch + Dinner)\n4. Roti Only\n\n"
        f'The user replied: "{user_input}"\n\n'
        f"Return ONLY one of these exact strings with no extra words:\n"
        f"Lunch\nDinner\nBoth\nRoti Only\n\n"
        f"If you cannot determine their choice, return ONLY: UNCLEAR"
    )

    response = llm.invoke(prompt, config=config)
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


def _parse_mis_issue_type(user_input: str) -> str | None:
    value = user_input.strip().lower()

    if value in {"1", "hardware", "hardware related"}:
        return "Hardware"

    if value in {"2", "operational", "operation", "operational related"}:
        return "Operational"

    if "hardware" in value:
        return "Hardware"

    if "operational" in value or "operation" in value:
        return "Operational"

    return None


# Node 2 — Query Reformulator


def query_reformulator(state: ChatState, config: RunnableConfig) -> dict:
    """
    Reformulate the user question into:
    1. A complete standalone query.
    2. One or more focused retrieval queries.
    """

    messages = state["messages"]
    current_question = messages[-1].content

    logger.info(f"Reformulating query: {current_question}")

    structured_llm = llm.with_structured_output(QueryReformulation)
    chain = reformulation_prompt | structured_llm

    response = chain.invoke(
        {
            "messages": messages[:-1],
            "question": current_question,
        },
        config=config,
    )

    standalone_query = response.standalone_query.strip()

    retrieval_queries = [
        query.strip() for query in response.retrieval_queries if query.strip()
    ]

    if not retrieval_queries:
        retrieval_queries = [standalone_query]

    logger.info(f"Reformulated query: {standalone_query}")
    logger.info(f"Retrieval queries: {retrieval_queries}")

    return {
        "standalone_query": standalone_query,
        "retrieval_queries": retrieval_queries,
        "web_search_used": False,
        "retrieval_attempts": 0,
        "grade_verdict": None,
        "grade_reason": None,
    }


# Node — Semantic Cache Lookup
def semantic_cache_lookup(state: ChatState, config: RunnableConfig) -> dict:
    """
    Checks the semantic cache for a previously-approved answer to a
    semantically similar standalone_query.

    Only ever called on the knowledge-lookup path (after guardrail has
    already routed intent to query_reformulator). Never runs for the
    stateful meal/mis/employee_request flows or conversational_response,
    since those bypass this node in the graph entirely.

    On hit: returns the cached answer as the AI message and sets
    cache_hit=True, so route_after_cache_lookup can route straight to END.

    On miss: sets cache_hit=False and lets the graph continue normally
    to adaptive_router.
    """
    query = state["standalone_query"]

    hit = check_semantic_cache(query)

    if hit is None:
        return {"cache_hit": False}

    return {
        "cache_hit": True,
        "generated_answer": hit["response"],
        "retrieved_docs": [],
        "web_search_used": False,
        "messages": [AIMessage(content=hit["response"])],
    }

# Node — Adaptive Router
def adaptive_router(state: ChatState, config: RunnableConfig) -> dict:
    """
    Classifies the reformulated query as simple or complex.

    Simple queries can skip CRAG document grading and proceed
    directly from retrieval to answer generation.

    Complex queries continue through the full CRAG
    grade -> rewrite -> re-retrieve workflow.
    """
    query = state["standalone_query"]

    logger.info(f"Adaptive routing query: {query}")

    structured_llm = llm.with_structured_output(QueryRoute)
    chain = adaptive_router_prompt | structured_llm

    result = chain.invoke(
        {
            "question": query,
        },
        config=config,
    )

    logger.info(f"Adaptive route: {result.route} | reason={result.reason}")

    return {
        "query_route": result.route,
        "routing_reason": result.reason,
    }


# Node 3 — Retriever


def retriever(state: ChatState, config: RunnableConfig) -> dict:
    """
    Hybrid search against Qdrant using one or more retrieval queries,
    with RAG Fusion query expansion and FlashRank reranking.

    For multi-intent questions, each retrieval query is searched
    independently, then results are merged and deduplicated.

    Relevance grading happens in grade_documents.
    """

    queries = state.get("retrieval_queries") or [state["standalone_query"]]

    filters = state.get("retrieval_filters")

    logger.info(f"Retrieving docs for {len(queries)} retrieval queries: {queries}")

    vector_store = get_vector_store()

    search_kwargs = {"k": settings.RETRIEVER_TOP_K}

    if filters:
        search_kwargs["filter"] = filters
        logger.info(f"Applying retrieval filters: {filters}")

    base_retriever = vector_store.as_retriever(
        search_kwargs=search_kwargs,
    )

    expanded_retriever = get_multi_query_retriever(base_retriever)

    reranking_retriever = get_reranking_retriever(expanded_retriever)

    all_docs = []

    # Retrieve independently for each decomposed query
    for query in queries:
        logger.info(f"Running retrieval query: {query}")

        docs = reranking_retriever.invoke(query, config=config)

        logger.info(f"Query returned {len(docs)} reranked chunks.")

        all_docs.extend(docs)

    # Deduplicate documents
    unique_docs = {}

    for doc in all_docs:
        doc_id = doc.metadata.get("_id") or (
            doc.metadata.get("file_name"),
            doc.metadata.get("page"),
            doc.page_content[:100],
        )

        # Keep the first occurrence.
        # Since each individual retrieval has already been reranked,
        # duplicates do not need to be added again.
        if doc_id not in unique_docs:
            unique_docs[doc_id] = doc

    docs = list(unique_docs.values())

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

    logger.info(
        f"Retrieved and merged → {len(serializable_docs)} unique chunks "
        f"from {len(queries)} retrieval queries."
    )

    return {"retrieved_docs": serializable_docs}


# Node — Grade Documents (CRAG)
def grade_documents(state: ChatState, config: RunnableConfig) -> dict:
    """
    Judges whether retrieved_docs are sufficient to answer standalone_query.
    Increments retrieval_attempts. Routing off this node's output decides
    whether to proceed to generation, loop back through a rewrite, or
    fall back to web search.
    """
    query = state["standalone_query"]
    docs = state.get("retrieved_docs", [])
    attempts = state.get("retrieval_attempts", 0) + 1

    if not docs:
        logger.info("CRAG grade: no documents retrieved.")
        verdict, reason = "irrelevant", "No documents were retrieved."
    else:
        context = "\n\n".join(d["page_content"] for d in docs)
        chain = get_grading_chain()
        result = chain.invoke({"question": query, "context": context}, config=config)
        verdict, reason = result.verdict, result.reason
        logger.info(f"CRAG grade (attempt {attempts}): {verdict} — {reason}")

    return {
        "grade_verdict": verdict,
        "grade_reason": reason,
        "retrieval_attempts": attempts,
    }


# Node — Rewrite Query (CRAG)
def rewrite_query(state: ChatState, config: RunnableConfig) -> dict:
    """
    Reformulates standalone_query after a failed grading or reflection,
    informed by the grader's stated reason for rejecting the previous retrieval.
    Loops back to retriever.
    """

    original_question = state["messages"][-1].content
    previous_query = state["standalone_query"]

    grade_reason = (
        state.get("reflection_feedback")
        or state.get("grade_reason")
        or "No relevant information found."
    )

    chain = rewrite_prompt | llm

    response = chain.invoke(
        {
            "original_question": original_question,
            "previous_query": previous_query,
            "grade_reason": grade_reason,
        },
        config=config,
    )

    new_query = response.content.strip()

    logger.info(
        f"CRAG rewrite (attempt {state.get('retrieval_attempts', 0)}): "
        f"'{previous_query}' → '{new_query}'"
    )

    return {
        "standalone_query": new_query,
        "retrieval_queries": [new_query],
        "query_route": (
            "complex" if state.get("reflection_feedback") else state.get("query_route")
        ),
        "reflection_action": None,
        "reflection_reason": None,
        "reflection_feedback": None,
    }


# Node 4 — Web Search
def web_search(state: ChatState, config: RunnableConfig) -> dict:
    """
    Fallback when RAG retrieval score is below threshold.
    Scopes search to NextBridge by injecting company name into query.
    Results formatted as retrieved_docs for answer_generator.
    """
    query = state["standalone_query"]
    scoped_query = f"NextBridge {query}"

    logger.info(f"Triggering web search for: {scoped_query}")

    tavily = get_tavily_search()
    results = tavily.invoke(scoped_query, config=config)

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

    return {"retrieved_docs": web_docs, "web_search_used": True}


# Node 5 — Answer Generator
def answer_generator(state: ChatState, config: RunnableConfig) -> dict:
    """
    Generates an answer using retrieved context + trimmed chat history.

    If reflection feedback exists, it is passed into the prompt so the
    regenerated answer can correct the issues identified by the critic.
    """

    context = format_context(state)
    trimmed_messages = trim_conversation(state)

    reflection_feedback = state.get("reflection_feedback")
    generation_attempts = state.get("generation_attempts", 0) + 1

    logger.info(f"Generating answer. Attempt #{generation_attempts}")

    chain = rag_prompt | llm

    invoke_payload = {
        "context": context,
        "messages": trimmed_messages,
        "reflection_feedback": reflection_feedback or "None",
    }

    response = chain.invoke(invoke_payload, config=config)

    answer_text = response.content.strip()

    logger.info("Answer generated.")

    return {
        "messages": [response],
        "generated_answer": answer_text,
        "generation_attempts": generation_attempts,
    }


def _get_latest_human_message(messages: list) -> str:
    """Returns the content of the most recent human message."""
    for msg in reversed(messages):
        if msg.__class__.__name__ == "HumanMessage":
            return msg.content
    return ""


# ---------------------------------------------------------------------------
# Node 6 — Meal subscription
# ---------------------------------------------------------------------------


def meal_subscription_node(state: ChatState, config: RunnableConfig) -> dict:
    """
    meal_data.step tracks where we are in the flow.
    route_entry bypasses guardrail so mid-flow messages reach this node directly.
    """
    meal = state.get("meal_data") or {}
    latest = _get_latest_human_message(state["messages"])

    if meal.get("email_sent"):
        return {
            "messages": [
                AIMessage(
                    content=(
                        f"Your **{meal.get('preference', 'meal')}** subscription is already submitted. "
                        f"Ask *'What is the status of my meal subscription?'* to check for updates."
                    )
                )
            ]
        }

    step = meal.get("step", "start")

    if step == "start":
        return {
            "meal_data": {**meal, "step": "waiting_preference", "in_progress": True},
            "messages": [AIMessage(content=MEAL_CHOICE_PROMPT)],
        }

    # ── Step 2: Parse meal preference — ask for name ────────────────────────
    if step == "waiting_preference":
        preference = _parse_meal_preference(latest, config=config)
        if not preference:
            return {
                "meal_data": {**meal},
                "messages": [AIMessage(content=MEAL_INVALID_PROMPT)],
            }
        return {
            "meal_data": {**meal, "step": "waiting_name", "preference": preference},
            "messages": [
                AIMessage(
                    content="Please enter your **full name** as it appears in HR records:"
                )
            ],
        }

    # ── Step 3: Save name — ask for employee ID ─────────────────────────────
    if step == "waiting_name":
        return {
            "meal_data": {**meal, "step": "waiting_emp_id", "name": latest.strip()},
            "messages": [
                AIMessage(content="Please enter your **Employee ID** (e.g. NXB-0042):")
            ],
        }

    # ── Step 4: Save emp ID — send email ────────────────────────────────────
    if step == "waiting_emp_id":
        emp_id = latest.strip()
        preference = meal.get("preference", "")
        name = meal.get("name", "")

        logger.info(f"Sending meal subscription: {name}, {emp_id}, {preference}")

        result = send_meal_subscription_email.invoke(
            {
                "name": name,
                "employee_id": emp_id,
                "preference": preference,
            },
            config=config,
        )

        tthread_id: str | None = None
        request_reference: str | None = None

        if "thread_id=" in result:
            thread_id_part = result.split("thread_id=", 1)[1]
            thread_id = thread_id_part.split(";", 1)[0].strip() or None

            if thread_id and thread_id.lower() == "none":
                thread_id = None

        if "request_reference=" in result:
            request_reference = result.split("request_reference=", 1)[1].strip() or None

        return {
            "meal_data": {
                **meal,
                "step": "completed",
                "employee_id": emp_id,
                "email_sent": True,
                "thread_id": thread_id,
                "request_reference": request_reference,
                "in_progress": False,
            },
            "messages": [
                AIMessage(
                    content=(
                        f"Done, **{name}**! Your **{preference}** subscription request "
                        f"(ID: {emp_id}) has been sent to the meals department.\n\n"
                        f"Ask *'What is the status of my meal subscription?'* anytime to check for a reply."
                    )
                )
            ],
        }

    return {
        "messages": [
            AIMessage(
                content="Something went wrong. Please say 'I want to subscribe to meals' to start again."
            )
        ]
    }


# Node 7 — Check meal subscription status


def check_meal_status_node(state: ChatState, config: RunnableConfig) -> dict:
    """
    Checks for a department reply.
    If reply found: shows it and asks for ack confirmation.
    If waiting_for_ack: processes yes/no from the latest message.
    """
    meal = state.get("meal_data") or {}
    latest = _get_latest_human_message(state["messages"])

    preference = meal.get("preference")
    name = meal.get("name", "the employee")
    emp_id = meal.get("employee_id", "N/A")
    thread_id = meal.get("thread_id")

    if not meal.get("email_sent") or not preference:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "I don't have a submitted subscription for this session. "
                        "Say *'I want to subscribe to meals'* to start one."
                    )
                )
            ]
        }
    if not thread_id:
        logger.warning("Meal status cannot be checked because no thread_id is stored.")
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Your meal request was sent, but its email tracking "
                        "information is unavailable. I cannot safely check its "
                        "status without risking showing an older request's reply."
                    )
                )
            ]
        }

    if meal.get("acknowledged"):
        return {
            "messages": [
                AIMessage(
                    content=f"Your **{preference}** subscription was already acknowledged. You're all set!"
                )
            ]
        }

    # ── Waiting for yes/no on acknowledgment ────────────────────────────────
    if meal.get("waiting_for_ack"):
        if latest.strip().lower() in ("yes", "y"):
            send_meal_acknowledgment.invoke(
                {
                    "name": name,
                    "employee_id": emp_id,
                },
                config=config,
            )
            return {
                "meal_data": {**meal, "acknowledged": True, "waiting_for_ack": False},
                "messages": [
                    AIMessage(
                        content="✅ Acknowledgment sent to the meals department. You're all set!"
                    )
                ],
            }
        return {
            "meal_data": {**meal, "waiting_for_ack": False},
            "messages": [
                AIMessage(
                    content="Okay, acknowledgment skipped. Ask for the status again anytime to send it."
                )
            ],
        }

    # ── Check for reply via @tool ────────────────────────────────────────────
    logger.info(f"Checking meal reply for thread_id={thread_id}")
    reply_body = check_meal_reply.invoke({"thread_id": thread_id}, config=config)

    if reply_body == "TRACKING_UNAVAILABLE":
        return {
            "messages": [
                AIMessage(
                    content=(
                        "I could not access the tracked email conversation for "
                        f"your **{preference}** subscription. Please try checking "
                        "again later."
                    )
                )
            ]
        }

    if reply_body == "NO_REPLY":
        return {
            "messages": [
                AIMessage(
                    content=(
                        "No reply yet from the meals department for your "
                        f"**{preference}** subscription. Please check back later."
                    )
                )
            ]
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
        "messages": [
            AIMessage(
                content=(
                    f"📬 The meals department has replied!\n\n"
                    f"**Their reply:** _{reply_body[:500]}_\n\n"
                    f"---\n"
                    f"**Draft acknowledgment:**\n```\n{ack_draft}\n```\n\n"
                    f"Should I send this? Reply **yes** to send or **no** to skip."
                )
            )
        ],
    }


def mis_request_node(state: ChatState, config: RunnableConfig) -> dict:
    mis = state.get("mis_data") or {}
    latest = _get_latest_human_message(state["messages"])

    if mis.get("email_sent"):
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Your MIS request has already been submitted. "
                        "Ask *'What is the status of my MIS request?'* "
                        "to check for updates."
                    )
                )
            ]
        }

    step = mis.get("step", "start")

    if step == "start":
        return {
            "mis_data": {
                **mis,
                "step": "waiting_issue_type",
                "in_progress": True,
            },
            "messages": [
                AIMessage(
                    content=(
                        "What type of MIS issue are you facing?\n\n"
                        "1. Hardware related\n"
                        "2. Operational"
                    )
                )
            ],
        }

    if step == "waiting_issue_type":
        issue_type = _parse_mis_issue_type(latest)

        if not issue_type:
            return {
                "mis_data": {**mis},
                "messages": [
                    AIMessage(
                        content=(
                            "Please select one of these options:\n\n"
                            "1. Hardware related\n"
                            "2. Operational"
                        )
                    )
                ],
            }

        return {
            "mis_data": {
                **mis,
                "step": "waiting_name",
                "issue_type": issue_type,
            },
            "messages": [
                AIMessage(
                    content=(
                        "Please enter your **full name** "
                        "as it appears in HR records:"
                    )
                )
            ],
        }

    if step == "waiting_name":
        return {
            "mis_data": {
                **mis,
                "step": "waiting_emp_id",
                "name": latest.strip(),
            },
            "messages": [
                AIMessage(content="Please enter your **Employee ID** (e.g. NXB-0042):")
            ],
        }

    if step == "waiting_emp_id":
        employee_id = latest.strip()
        name = mis.get("name", "")
        issue_type = mis.get("issue_type", "")

        result = send_mis_request_email.invoke(
            {
                "issue_type": issue_type,
                "name": name,
                "employee_id": employee_id,
            },
            config=config,
        )

        thread_id: str | None = None
        request_reference: str | None = None

        if "thread_id=" in result:
            thread_id = (
                result.split("thread_id=", 1)[1].split(";", 1)[0].strip() or None
            )

            if thread_id and thread_id.lower() == "none":
                thread_id = None

        if "request_reference=" in result:
            request_reference = result.split("request_reference=", 1)[1].strip() or None

        return {
            "mis_data": {
                **mis,
                "step": "completed",
                "employee_id": employee_id,
                "email_sent": True,
                "thread_id": thread_id,
                "request_reference": request_reference,
                "in_progress": False,
            },
            "messages": [
                AIMessage(
                    content=(
                        f"Done, **{name}**! Your **{issue_type}** MIS request "
                        f"(ID: {employee_id}) has been sent to the MIS team.\n\n"
                        "Ask *'What is the status of my MIS request?'* "
                        "anytime to check for a reply."
                    )
                )
            ],
        }

    return {
        "messages": [
            AIMessage(
                content=(
                    "Something went wrong. Please say "
                    "'I want to contact MIS' to start again."
                )
            )
        ]
    }


def check_mis_status_node(state: ChatState, config: RunnableConfig) -> dict:
    mis = state.get("mis_data") or {}
    latest = _get_latest_human_message(state["messages"])

    issue_type = mis.get("issue_type")
    name = mis.get("name", "the employee")
    employee_id = mis.get("employee_id", "N/A")
    thread_id = mis.get("thread_id")

    if not mis.get("email_sent") or not issue_type:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "I don't have a submitted MIS request for this session. "
                        "Say *'I want to contact MIS'* to create one."
                    )
                )
            ]
        }

    if not thread_id:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Your MIS request was sent, but its email tracking "
                        "information is unavailable. I cannot safely check its "
                        "status without risking showing an older request's reply."
                    )
                )
            ]
        }

    if mis.get("acknowledged"):
        return {
            "messages": [
                AIMessage(content="Your MIS response has already been acknowledged.")
            ]
        }

    if mis.get("waiting_for_ack"):
        if latest.strip().lower() in {"yes", "y"}:
            send_mis_acknowledgment.invoke(
                {
                    "name": name,
                    "employee_id": employee_id,
                },
                config=config,
            )

            return {
                "mis_data": {
                    **mis,
                    "acknowledged": True,
                    "waiting_for_ack": False,
                },
                "messages": [AIMessage(content="Acknowledgment sent to the MIS team.")],
            }

        return {
            "mis_data": {
                **mis,
                "waiting_for_ack": False,
            },
            "messages": [
                AIMessage(
                    content=(
                        "Okay, acknowledgment skipped. "
                        "Ask for the status again anytime."
                    )
                )
            ],
        }

    reply_body = check_mis_reply.invoke({"thread_id": thread_id}, config=config)

    if reply_body == "TRACKING_UNAVAILABLE":
        return {
            "messages": [
                AIMessage(
                    content=(
                        "I could not access the tracked email conversation "
                        "for your MIS request. Please check again later."
                    )
                )
            ]
        }

    if reply_body == "NO_REPLY":
        return {
            "messages": [
                AIMessage(
                    content=(
                        "No reply yet from the MIS team for your "
                        f"**{issue_type}** request. Please check back later."
                    )
                )
            ]
        }

    acknowledgment = (
        f"Dear MIS Team,\n\n"
        f"Thank you for your response regarding the MIS request for "
        f"{name} (ID: {employee_id}).\n\n"
        f"We acknowledge receipt and will act accordingly.\n\n"
        f"Regards,\nNXB Chatbot System"
    )

    return {
        "mis_data": {
            **mis,
            "waiting_for_ack": True,
        },
        "messages": [
            AIMessage(
                content=(
                    "📬 The MIS team has replied!\n\n"
                    f"**Their reply:** _{reply_body[:500]}_\n\n"
                    "---\n"
                    "**Draft acknowledgment:**\n"
                    f"```\n{acknowledgment}\n```\n\n"
                    "Should I send this? Reply **yes** to send "
                    "or **no** to skip."
                )
            )
        ],
    }


# ---------------------------------------------------------------------------
# Leave / Work From Home autonomous request node
# ---------------------------------------------------------------------------


def employee_request_node(state: ChatState, config: RunnableConfig) -> dict:
    """
    Autonomous LLM-driven Leave / Work From Home conversation.

    The LLM:
    - extracts values;
    - understands corrections;
    - chooses the next question;
    - creates confirmation summaries;
    - interprets confirmation.

    Python executes the external Gmail action.
    """
    request_data = state.get("employee_request_data") or {}
    latest = _get_latest_human_message(state["messages"])

    if request_data.get("email_sent"):
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Your request has already been submitted to the "
                        "General Manager. Ask for the status of your leave "
                        "or work-from-home request to check for a reply."
                    )
                )
            ]
        }

    # The employee was already shown a complete summary.
    if request_data.get("confirmation_requested"):
        confirmation_chain = employee_confirmation_prompt | llm.with_structured_output(
            EmployeeConfirmationDecision
        )

        confirmation = confirmation_chain.invoke(
            {
                "latest_message": latest,
            },
            config=config,
        )

        if confirmation.action == "confirmed":
            required_fields = (
                "request_type",
                "employee_name",
                "employee_id",
                "start_date",
                "end_date",
            )

            missing_fields = [
                field for field in required_fields if not request_data.get(field)
            ]

            if missing_fields:
                # Defensive fallback: return control to the request LLM.
                request_data = {
                    **request_data,
                    "confirmation_requested": False,
                }
            else:
                tool_result = send_employee_request_to_gm.invoke(
                    {
                        "request_type": request_data["request_type"],
                        "employee_name": request_data["employee_name"],
                        "employee_id": request_data["employee_id"],
                        "start_date": request_data["start_date"],
                        "end_date": request_data["end_date"],
                        "reason": request_data.get("reason", ""),
                    },
                    config=config,
                )

                if tool_result.startswith("ERROR:"):
                    return {
                        "employee_request_data": {
                            **request_data,
                            "confirmation_requested": False,
                        },
                        "messages": [
                            AIMessage(
                                content=(
                                    "I could not submit the request because "
                                    f"the email operation failed: {tool_result}"
                                )
                            )
                        ],
                    }

                thread_id, request_reference = _extract_tracking_data(tool_result)

                return {
                    "employee_request_data": {
                        **request_data,
                        "confirmation_requested": False,
                        "email_sent": True,
                        "in_progress": False,
                        "thread_id": thread_id,
                        "request_reference": request_reference,
                        "waiting_for_ack": False,
                        "acknowledged": False,
                    },
                    "messages": [
                        AIMessage(
                            content=(
                                f"Your **{request_data['request_type']}** "
                                "request has been sent to the General Manager. "
                                "You can ask me for its status when you want "
                                "to check for a reply."
                            )
                        )
                    ],
                }

        elif confirmation.action == "rejected":
            return {
                "employee_request_data": {},
                "messages": [AIMessage(content=confirmation.response)],
            }

        elif confirmation.action == "correction":
            # Let the main extraction LLM read the correction and update fields.
            request_data = {
                **request_data,
                "confirmation_requested": False,
            }

        else:
            return {
                "employee_request_data": request_data,
                "messages": [AIMessage(content=confirmation.response)],
            }

    request_chain = employee_request_prompt | llm.with_structured_output(
        EmployeeRequestDecision
    )

    decision = request_chain.invoke(
        {
            "current_date": date.today().isoformat(),
            "request_data": json.dumps(
                _employee_request_view(request_data),
                ensure_ascii=False,
            ),
            "latest_message": latest,
            "messages": state["messages"],
        },
        config=config,
    )

    update_values = decision.model_dump()

    collected_fields = {
        "request_type",
        "employee_name",
        "employee_id",
        "start_date",
        "end_date",
        "reason",
    }

    updated_request = _merge_non_null_values(
        request_data,
        update_values,
        collected_fields,
    )

    if decision.action == "cancel_request":
        return {
            "employee_request_data": {},
            "messages": [AIMessage(content=decision.response)],
        }

    if decision.action == "request_confirmation":
        return {
            "employee_request_data": {
                **updated_request,
                "in_progress": True,
                "confirmation_requested": True,
            },
            "messages": [AIMessage(content=decision.response)],
        }

    if decision.action == "send_request":
        # The LLM is not allowed to bypass explicit confirmation.
        # Convert an early send decision into a confirmation request.
        return {
            "employee_request_data": {
                **updated_request,
                "in_progress": True,
                "confirmation_requested": True,
            },
            "messages": [
                AIMessage(
                    content=(
                        f"{decision.response}\n\n"
                        "Please confirm clearly whether I should send this "
                        "request to the General Manager."
                    )
                )
            ],
        }

    return {
        "employee_request_data": {
            **updated_request,
            "in_progress": True,
        },
        "messages": [AIMessage(content=decision.response)],
    }


def check_employee_request_status_node(
    state: ChatState, config: RunnableConfig
) -> dict:
    """
    Checks the GM response and uses the LLM to create and manage a
    context-aware acknowledgement.
    """
    request_data = state.get("employee_request_data") or {}
    latest = _get_latest_human_message(state["messages"])

    if not request_data.get("email_sent"):
        return {
            "messages": [
                AIMessage(
                    content=(
                        "I do not have a submitted Leave or Work From Home "
                        "request for this conversation."
                    )
                )
            ]
        }

    thread_id = request_data.get("thread_id")

    if not thread_id:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Your request was sent, but its Gmail tracking "
                        "information is unavailable, so I cannot safely "
                        "check the corresponding GM reply."
                    )
                )
            ]
        }

    if request_data.get("acknowledged"):
        return {
            "messages": [
                AIMessage(
                    content=(
                        "The General Manager's response has already been "
                        "acknowledged."
                    )
                )
            ]
        }

    # Employee has already been shown an acknowledgement draft.
    if request_data.get("waiting_for_ack"):
        confirmation_chain = (
            acknowledgement_confirmation_prompt
            | llm.with_structured_output(AcknowledgementConfirmationDecision)
        )

        confirmation = confirmation_chain.invoke(
            {
                "latest_message": latest,
            },
            config=config,
        )

        if confirmation.action == "send":
            acknowledgement = request_data.get(
                "acknowledgement_draft",
                "",
            )

            result = send_gm_employee_request_acknowledgement.invoke(
                {
                    "thread_id": thread_id,
                    "acknowledgement": acknowledgement,
                },
                config=config,
            )

            if result.startswith("ERROR:"):
                return {
                    "employee_request_data": request_data,
                    "messages": [
                        AIMessage(
                            content=(
                                "The acknowledgement could not be sent. " f"{result}"
                            )
                        )
                    ],
                }

            return {
                "employee_request_data": {
                    **request_data,
                    "waiting_for_ack": False,
                    "acknowledged": True,
                },
                "messages": [AIMessage(content=confirmation.response)],
            }

        if confirmation.action == "skip":
            return {
                "employee_request_data": {
                    **request_data,
                    "waiting_for_ack": False,
                },
                "messages": [AIMessage(content=confirmation.response)],
            }

        if confirmation.action == "regenerate":
            gm_reply = request_data.get("gm_reply", "")

            acknowledgement_chain = (
                gm_acknowledgement_prompt
                | llm.with_structured_output(GMAcknowledgementResult)
            )

            generated = acknowledgement_chain.invoke(
                {
                    "request_data": json.dumps(
                        _employee_request_view(request_data),
                        ensure_ascii=False,
                    ),
                    "gm_reply": gm_reply,
                    "employee_name": request_data.get(
                        "employee_name",
                        "Employee",
                    ),
                    "employee_id": request_data.get(
                        "employee_id",
                        "N/A",
                    ),
                },
                config=config,
            )

            return {
                "employee_request_data": {
                    **request_data,
                    "acknowledgement_draft": (generated.acknowledgement),
                    "waiting_for_ack": True,
                },
                "messages": [
                    AIMessage(
                        content=(
                            f"{generated.reply_summary}\n\n"
                            "**Rewritten acknowledgement:**\n"
                            f"```\n{generated.acknowledgement}\n```\n\n"
                            "Should I send it to the General Manager?"
                        )
                    )
                ],
            }

        return {
            "employee_request_data": request_data,
            "messages": [AIMessage(content=confirmation.response)],
        }

    gm_reply = check_gm_employee_request_reply.invoke(
        {
            "thread_id": thread_id,
        },
        config=config,
    )

    if gm_reply == "TRACKING_UNAVAILABLE":
        return {
            "messages": [
                AIMessage(
                    content=(
                        "I could not access the tracked Gmail conversation. "
                        "Please check the status again later."
                    )
                )
            ]
        }

    if gm_reply == "NO_REPLY":
        return {
            "messages": [
                AIMessage(
                    content=(
                        "The General Manager has not replied to your "
                        f"**{request_data.get('request_type', 'request')}** "
                        "request yet."
                    )
                )
            ]
        }

    acknowledgement_chain = gm_acknowledgement_prompt | llm.with_structured_output(
        GMAcknowledgementResult
    )

    generated = acknowledgement_chain.invoke(
        {
            "request_data": json.dumps(
                _employee_request_view(request_data),
                ensure_ascii=False,
            ),
            "gm_reply": gm_reply,
            "employee_name": request_data.get(
                "employee_name",
                "Employee",
            ),
            "employee_id": request_data.get(
                "employee_id",
                "N/A",
            ),
        },
        config=config,
    )

    return {
        "employee_request_data": {
            **request_data,
            "gm_reply": gm_reply,
            "acknowledgement_draft": generated.acknowledgement,
            "waiting_for_ack": True,
        },
        "messages": [
            AIMessage(
                content=(
                    f"📬 **The General Manager has replied.**\n\n"
                    f"{generated.reply_summary}\n\n"
                    f"**GM reply:**\n"
                    f"> {gm_reply[:1000]}\n\n"
                    f"**LLM-generated acknowledgement:**\n"
                    f"```\n{generated.acknowledgement}\n```\n\n"
                    "Should I send this acknowledgement?"
                )
            )
        ],
    }


def reflect_answer(state: ChatState, config: RunnableConfig) -> dict:
    """
    Critique the generated answer against the retrieved context
    and decide whether to pass, regenerate, or retrieve again.
    """

    question = state["standalone_query"]
    answer = state.get("generated_answer") or ""

    docs = state.get("retrieved_docs", [])

    context = "\n\n".join(
        doc.get("page_content", "") for doc in docs if doc.get("page_content")
    )

    logger.info("Reflecting on generated answer.")
    structured_llm = llm.with_structured_output(AnswerReflection)
    chain = reflection_prompt | structured_llm

    result = chain.invoke(
        {
            "question": question,
            "context": context,
            "answer": answer,
        },
        config=config,
    )

    reflection_attempts = state.get("reflection_attempts", 0) + 1

    logger.info(
        f"Reflection result → "
        f"action={result.action}, "
        f"grounded={result.grounded}, "
        f"complete={result.complete}, "
        f"relevant={result.relevant}"
    )

    logger.info(f"Reflection feedback: {result.feedback}")

    if result.action == "pass" and not state.get("web_search_used"):
        store_semantic_cache(
            query=question,
            answer=answer,
            retrieved_docs=docs,
        )

    return {
        "reflection_action": result.action,
        "reflection_reason": (
            f"grounded={result.grounded}, "
            f"complete={result.complete}, "
            f"relevant={result.relevant}"
        ),
        "reflection_feedback": result.feedback,
        "reflection_attempts": reflection_attempts,
    }