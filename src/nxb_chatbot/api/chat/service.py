import logging
import uuid

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
    # Get or create session
    if request.session_id:
        session = await get_session_by_id(request.session_id, db)
        logger.info(f"Resuming session: {session.id}")
    else:
        session = await create_session(db)
        logger.info(f"New session: {session.id}")

    # LangGraph config — thread_id ties to checkpointer
    config = {"configurable": {"thread_id": session.thread_id}}

    try:
        state = await graph.ainvoke(
            {
                "messages": [HumanMessage(content=request.message)],
                "retrieved_docs": [],
                "retrieval_filters": request.retrieval_filters,
                "standalone_query": None,
            },
            config=config,
        )
    except Exception as e:
        logger.error(f"Graph invocation failed: {e}")
        raise GraphInvokeException(str(e))

    answer = state["messages"][-1].content
    retrieved_docs = state.get("retrieved_docs", [])

    return ChatResponse(
        session_id=session.id,
        thread_id=session.thread_id,
        answer=answer,
        retrieved_docs=retrieved_docs,
    )


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