from fastapi import APIRouter

from nxb_chatbot.api.ingest.schema import IngestResponse
from nxb_chatbot.api.ingest.service import run_ingestion
from nxb_chatbot.api.deps import DBSession

router = APIRouter(prefix="/ingest", tags=["Ingest"])


@router.post("/", response_model=IngestResponse)
async def ingest_endpoint(db: DBSession) -> IngestResponse:
    return await run_ingestion(db)