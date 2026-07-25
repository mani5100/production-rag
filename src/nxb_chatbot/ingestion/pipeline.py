import logging
from pathlib import Path
from nxb_chatbot.core.config import settings

from nxb_chatbot.ingestion.loaders import load_pdf
from nxb_chatbot.ingestion.splitters import split_documents
from nxb_chatbot.vector_store.qdrant_client import upsert_documents

logger = logging.getLogger(__name__)
DATA_FOLDER = Path(settings.DATA_FOLDER)


def run_ingestion_pipeline() -> None:
    """
    Main ingestion pipeline. Scans data/ folder and processes each PDF:
        1. Load PDF → list of Documents (per page)
        2. Split Documents → chunks (tables atomic, text split)
        3. Upsert chunks → Qdrant (with duplicate + change detection)

    One failed PDF does not stop the rest.
    """
    if not DATA_FOLDER.exists():
        raise FileNotFoundError(f"Data folder not found: {DATA_FOLDER.resolve()}")

    pdf_files = sorted(DATA_FOLDER.glob("*.pdf"))

    if not pdf_files:
        logger.warning("No PDF files found in data/ folder.")
        return

    logger.info(f"Starting ingestion for {len(pdf_files)} PDFs.")

    success, skipped, failed = 0, 0, 0

    for pdf_file in pdf_files:
        try:
            logger.info(f"Processing: {pdf_file.name}")

            # Step 1 — Load
            documents = load_pdf(pdf_file)
            if not documents:
                logger.warning(f"No content extracted from '{pdf_file.name}'. Skipping.")
                skipped += 1
                continue

            # Step 2 — Split
            chunks = split_documents(documents)
            if not chunks:
                logger.warning(f"No chunks produced for '{pdf_file.name}'. Skipping.")
                skipped += 1
                continue

            # Step 3 — Upsert (duplicate + change detection inside)
            upsert_documents(chunks, pdf_file)
            success += 1

        except Exception as e:
            logger.error(f"Failed to process '{pdf_file.name}': {e}")
            failed += 1

    logger.info(
        f"Ingestion complete — "
        f"Success: {success} | Skipped: {skipped} | Failed: {failed}"
    )