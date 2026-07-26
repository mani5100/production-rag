import uuid
from datetime import datetime

from pydantic import BaseModel


class IngestResponse(BaseModel):
    message: str
    total_pdfs: int
    success: int
    skipped: int
    failed: int
    documents: list["DocumentResponse"]


class DocumentResponse(BaseModel):
    id: uuid.UUID
    file_name: str
    file_hash: str
    page_count: int
    chunk_count: int
    status: str
    error_message: str | None
    created_at: datetime

    model_config = {"from_attributes": True}