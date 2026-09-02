import logging
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nxb_chatbot.api.ingest.exceptions import IngestionFailedException
from nxb_chatbot.api.ingest.model import IngestedDocument
from nxb_chatbot.api.ingest.schema import DocumentResponse, IngestResponse
from nxb_chatbot.core.config import settings
from nxb_chatbot.ingestion.loaders import load_pdf
from nxb_chatbot.ingestion.pipeline import _save_chunks_for_inspection
from nxb_chatbot.ingestion.splitters import split_documents
from nxb_chatbot.vector_store.qdrant_client import (
    compute_file_hash,
    get_existing_hash,
    get_qdrant_client,
    upsert_documents,
)

logger = logging.getLogger(__name__)

DATA_FOLDER = Path(settings.DATA_FOLDER)


async def run_ingestion(db: AsyncSession) -> IngestResponse:
    if not DATA_FOLDER.exists():
        raise IngestionFailedException(f"Data folder not found: {DATA_FOLDER.resolve()}")

    pdf_files = sorted(DATA_FOLDER.glob("*.pdf"))

    if not pdf_files:
        raise IngestionFailedException("No PDF files found in data folder.")

    success, skipped, failed = 0, 0, 0
    document_responses = []
    client = get_qdrant_client()

    for pdf_file in pdf_files:
        # Check if already ingested with same hash
        file_hash = compute_file_hash(pdf_file)
        try:
            existing_hash = get_existing_hash(client, pdf_file.name)
        except RuntimeError as e:
            raise IngestionFailedException(str(e))
        
        # Get or create DB record
        result = await db.execute(
            select(IngestedDocument).where(
                IngestedDocument.file_name == pdf_file.name
            )
        )
        doc_record = result.scalar_one_or_none()

        if existing_hash == file_hash:
            logger.info(f"Skipping '{pdf_file.name}' — no changes.")
            skipped += 1
            if doc_record:
                document_responses.append(
                    DocumentResponse.model_validate(doc_record)
                )
            continue

        # Create or update DB record
        if not doc_record:
            doc_record = IngestedDocument(
                file_name=pdf_file.name,
                file_hash=file_hash,
                status="processing",
            )
            db.add(doc_record)
        else:
            doc_record.file_hash = file_hash
            doc_record.status = "processing"
            doc_record.error_message = None

        await db.flush()

        try:
            documents = load_pdf(pdf_file)
            chunks = split_documents(documents)

            _save_chunks_for_inspection(chunks, pdf_file)

            upsert_documents(chunks, pdf_file)

            doc_record.status = "success"
            doc_record.page_count = len(documents)
            doc_record.chunk_count = len(chunks)

            success += 1
            logger.info(f"Ingested '{pdf_file.name}' — {len(chunks)} chunks.")

        except Exception as e:
            doc_record.status = "failed"
            doc_record.error_message = str(e)
            failed += 1
            logger.error(f"Failed to ingest '{pdf_file.name}': {e}")

        await db.flush()
        document_responses.append(DocumentResponse.model_validate(doc_record))

    return IngestResponse(
        message="Ingestion complete.",
        total_pdfs=len(pdf_files),
        success=success,
        skipped=skipped,
        failed=failed,
        documents=document_responses,
    )