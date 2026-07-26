import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User message to the chatbot",
    )
    session_id: uuid.UUID | None = Field(
        default=None,
        description="Existing session ID. If None, a new session is created.",
    )
    retrieval_filters: dict | None = Field(
        default=None,
        description="Optional Qdrant metadata filters e.g. {'file_name': 'policy.pdf'}",
    )


class ChatResponse(BaseModel):
    session_id: uuid.UUID
    thread_id: str
    answer: str
    retrieved_docs: list[dict]


class SessionResponse(BaseModel):
    id: uuid.UUID
    thread_id: str
    created_at: datetime

    model_config = {"from_attributes": True}