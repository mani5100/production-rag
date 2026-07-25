import logging

from nxb_chatbot.rag.prompts import rag_prompt, reformulation_prompt
from nxb_chatbot.rag.reranker import get_reranking_retriever
from nxb_chatbot.rag.services import format_context, llm, trim_conversation
from nxb_chatbot.rag.state import ChatState
from nxb_chatbot.vector_store.qdrant_client import get_vector_store
from nxb_chatbot.core.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node 1 — Query Reformulator
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Node 2 — Retriever
# ---------------------------------------------------------------------------
def retriever(state: ChatState) -> dict:
    """
    Hybrid search against Qdrant using standalone_query.
    Applies metadata filters if set in state.
    Reranks results before returning.
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

    serializable_docs = [
        {
            "page_content": doc.page_content,
            "metadata": {
                **doc.metadata,
                "relevance_score": float(doc.metadata.get("relevance_score", 0.0)),
            },
        }
        for doc in docs
    ]

    logger.info(f"Retrieved and reranked → {len(serializable_docs)} final chunks.")
    return {"retrieved_docs": serializable_docs}

# ---------------------------------------------------------------------------
# Node 3 — Answer Generator
# ---------------------------------------------------------------------------

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