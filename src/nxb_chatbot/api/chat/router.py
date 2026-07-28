import logging
import uuid

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from nxb_chatbot.api.chat.schema import ChatRequest, ChatResponse, SessionResponse
from nxb_chatbot.api.chat.service import (
    chat,
    get_session_messages,
    list_sessions,
    stream_chat,
)
from nxb_chatbot.api.deps import DBSession, GraphDep

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    graph: GraphDep,
    db: DBSession,
) -> ChatResponse:
    return await chat(request, graph, db)


@router.get("/sessions", response_model=list[SessionResponse])
async def get_sessions(db: DBSession) -> list[SessionResponse]:
    return await list_sessions(db)

@router.get("/sessions/{session_id}/messages", response_model=list[dict])
async def get_messages_endpoint(
    session_id: uuid.UUID,
    graph: GraphDep,
    db: DBSession,
) -> list[dict]:
    return await get_session_messages(session_id, graph, db)

@router.post("/stream")
async def stream_chat_endpoint(
    request: ChatRequest,
    graph: GraphDep,
    db: DBSession,
) -> StreamingResponse:
    return StreamingResponse(
        stream_chat(
            request=request,
            graph=graph,
            db=db,
        ),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )