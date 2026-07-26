from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from nxb_chatbot.db.base import BaseModel


class ChatSession(BaseModel):
    __tablename__ = "chat_sessions"

    thread_id: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )