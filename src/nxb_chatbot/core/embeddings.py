import logging

from langchain_ollama import OllamaEmbeddings

from nxb_chatbot.core.config import settings

logger = logging.getLogger(__name__)


def get_dense_embedder() -> OllamaEmbeddings:
    """
    Returns the Ollama dense embedding model.

    Used by both the ingestion pipeline and the RAG retriever.
    """
    logger.info(
        "Loading embedding model: %s from %s",
        settings.EMBEDDING_MODEL,
        settings.OLLAMA_BASE_URL,
    )

    return OllamaEmbeddings(
        model=settings.EMBEDDING_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
    )