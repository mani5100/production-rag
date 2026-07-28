import logging
import uuid

import json
from collections.abc import AsyncGenerator

from langgraph.types import Command
from langchain_core.messages import HumanMessage
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nxb_chatbot.api.chat.exceptions import GraphInvokeException, SessionNotFoundException
from nxb_chatbot.api.chat.model import ChatSession
from nxb_chatbot.api.chat.schema import ChatRequest, ChatResponse, SessionResponse

logger = logging.getLogger(__name__)

# Session Helpers

async def get_session_by_id(
    session_id: uuid.UUID,
    db: AsyncSession,
) -> ChatSession:
    result = await db.execute(
        select(ChatSession).where(ChatSession.id == session_id)
    )
    session = result.scalar_one_or_none()
    if not session:
        raise SessionNotFoundException(str(session_id))
    return session


async def create_session(db: AsyncSession) -> ChatSession:
    session = ChatSession(thread_id=str(uuid.uuid4()))
    db.add(session)
    await db.flush()
    logger.info(f"Created new chat session: {session.id} | thread: {session.thread_id}")
    return session


# Chat
async def chat(
    request: ChatRequest,
    graph,
    db: AsyncSession,
) -> ChatResponse:
    if request.session_id:
        session = await get_session_by_id(request.session_id, db)
        logger.info(f"Resuming session: {session.id}")
    else:
        session = await create_session(db)
        logger.info(f"New session: {session.id}")

    config = {"configurable": {"thread_id": session.thread_id}}

    input_data = {
        "messages": [HumanMessage(content=request.message)],
        "retrieved_docs": [],
        "retrieval_filters": request.retrieval_filters,
        "standalone_query": None,
        "guardrail_passed": None,
        "web_search_used": None,
        "meal_intent": None,
        # meal_data excluded — preserved from checkpoint across turns
    }

    try:
        state = await graph.ainvoke(input_data, config=config)
    except Exception as e:
        logger.error(f"Graph invocation failed: {e}", exc_info=True)
        raise GraphInvokeException(str(e))

    answer = state["messages"][-1].content

    return ChatResponse(
        session_id=session.id,
        thread_id=session.thread_id,
        answer=answer,
        retrieved_docs=state.get("retrieved_docs", []),
        web_search_used=state.get("web_search_used"),
        guardrail_passed=state.get("guardrail_passed"),
    )
    
    
async def stream_chat(
    request: ChatRequest,
    graph,
    db: AsyncSession,
) -> AsyncGenerator[str, None]:
    """Run LangGraph and stream events as newline-delimited JSON."""

    if request.session_id:
        session = await get_session_by_id(
            request.session_id,
            db,
        )
        logger.info("Resuming streaming session: %s", session.id)
    else:
        session = await create_session(db)
        logger.info("Created streaming session: %s", session.id)

    config = {
        "configurable": {
            "thread_id": session.thread_id,
        }
    }

    input_data = {
        "messages": [
            HumanMessage(content=request.message)
        ],
        "retrieved_docs": [],
        "retrieval_filters": request.retrieval_filters,
        "standalone_query": None,
        "guardrail_passed": None,
        "web_search_used": None,
        "meal_intent": None,
    }

    # Send the session information immediately.
    yield json.dumps(
        {
            "type": "session",
            "session_id": str(session.id),
            "thread_id": session.thread_id,
        }
    ) + "\n"

    try:
        async for message_chunk, metadata in graph.astream(
            input_data,
            config=config,
            stream_mode="messages",
        ):
            node_name = metadata.get("langgraph_node")

            # Only stream the final RAG answer.
            # Otherwise tokens from guardrail and query reformulation
            # may also appear in the UI.
            if node_name != "answer_generator":
                continue

            content = getattr(message_chunk, "content", "")

            if isinstance(content, str) and content:
                yield json.dumps(
                    {
                        "type": "token",
                        "content": content,
                    }
                ) + "\n"

        # Read the completed checkpointed state.
        final_state = await graph.aget_state(config)

        values = final_state.values if final_state else {}

        yield json.dumps(
            {
                "type": "done",
                "session_id": str(session.id),
                "thread_id": session.thread_id,
                "retrieved_docs": values.get(
                    "retrieved_docs",
                    [],
                ),
                "web_search_used": values.get(
                    "web_search_used",
                    False,
                ),
                "guardrail_passed": values.get(
                    "guardrail_passed",
                ),
            }
        ) + "\n"

    except Exception as exc:
        logger.error(
            "Streaming graph invocation failed: %s",
            exc,
            exc_info=True,
        )

        yield json.dumps(
            {
                "type": "error",
                "message": str(exc),
            }
        ) + "\n"
    

# List Sessions
async def list_sessions(db: AsyncSession) -> list[SessionResponse]:
    result = await db.execute(
        select(ChatSession).order_by(ChatSession.created_at.desc())
    )
    sessions = result.scalars().all()
    return [SessionResponse.model_validate(s) for s in sessions]


async def get_session_messages(
    session_id: uuid.UUID,
    graph,
    db: AsyncSession,
) -> list[dict]:
    # Verify session exists in our DB
    session = await get_session_by_id(session_id, db)

    # Fetch messages from LangGraph state
    config = {"configurable": {"thread_id": session.thread_id}}
    state = await graph.aget_state(config)

    if not state or not state.values:
        return []

    messages = state.values.get("messages", [])

    return [
        {
            "role": "human" if msg.__class__.__name__ == "HumanMessage" else "assistant",
            "content": msg.content,
        }
        for msg in messages
    ]