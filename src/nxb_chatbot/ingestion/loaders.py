import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_pymupdf4llm import PyMuPDF4LLMLoader

logger = logging.getLogger(__name__)

from nxb_chatbot.core.config import settings

DATA_FOLDER = Path(settings.DATA_FOLDER)


def load_pdf(file_path: Path) -> list[Document]:
    try:
        loader = PyMuPDF4LLMLoader(
            file_path=str(file_path),
            mode="page",
        )
        documents = loader.load()

        # Enrich metadata — loader does not set these by default
        for doc in documents:
            doc.metadata["file_name"] = file_path.name
            # Convert 0-indexed to 1-indexed page numbers
            if "page" in doc.metadata:
                doc.metadata["page"] = doc.metadata["page"] + 1

        logger.info(f"Loaded {len(documents)} pages from {file_path.name}")
        return documents

    except Exception as e:
        logger.error(f"Failed to load {file_path.name}: {e}")
        return []


def load_all_pdfs() -> list[Document]:
    if not DATA_FOLDER.exists():
        raise FileNotFoundError(f"Data folder not found: {DATA_FOLDER.resolve()}")

    pdf_files = sorted(DATA_FOLDER.glob("*.pdf"))

    if not pdf_files:
        logger.warning("No PDF files found in data/ folder")
        return []

    logger.info(f"Found {len(pdf_files)} PDFs in {DATA_FOLDER.resolve()}")

    all_documents = []
    for pdf_file in pdf_files:
        docs = load_pdf(pdf_file)
        all_documents.extend(docs)

    logger.info(f"Total pages loaded: {len(all_documents)}")
    return all_documents