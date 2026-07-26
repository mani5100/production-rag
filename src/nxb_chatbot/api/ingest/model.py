from sqlalchemy import Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from nxb_chatbot.db.base import BaseModel


class IngestedDocument(BaseModel):
    __tablename__ = "ingested_documents"

    file_name: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )
    file_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    page_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    chunk_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="processing",
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )