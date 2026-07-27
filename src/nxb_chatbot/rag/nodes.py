import logging

from langchain_core.messages import AIMessage

from nxb_chatbot.core.config import settings
from nxb_chatbot.rag.prompts import rag_prompt, reformulation_prompt
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

logger = logging.getLogger(__name__)

OFF_TOPIC_RESPONSE = (
    "I can only answer questions related to NextBridge Ltd. "
    "Please ask about company policies, procedures, or internal matters."
)



# Node 1 — Guardrail

def guardrail(state: ChatState) -> dict:
    """
    Classifies whether the query is NextBridge related.
    Sets guardrail_passed in state.
    If failed — injects canned response directly into messages.
    """
    messages = state["messages"]
    question = messages[-1].content

    logger.info(f"Running guardrail check for: {question}")

    chain = get_guardrail_chain()
    result: GuardrailResult = chain.invoke({"question": question})

    logger.info(
        f"Guardrail result: passed={result.passed} | reason={result.reason}"
    )

    if not result.passed:
        return {
            "guardrail_passed": False,
            "messages": [AIMessage(content=OFF_TOPIC_RESPONSE)],
        }

    return {"guardrail_passed": True}


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