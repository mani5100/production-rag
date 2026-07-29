import logging

from langchain_huggingface import HuggingFaceEmbeddings

from nxb_chatbot.core.config import settings

logger = logging.getLogger(__name__)


def get_dense_embedder() -> HuggingFaceEmbeddings:
    """
    Returns the local Hugging Face dense embedding model.

    Used by both the ingestion pipeline and the RAG retriever.
    """
    logger.info(
        "Loading embedding model: %s on %s",
        settings.EMBEDDING_MODEL,
        settings.EMBEDDING_DEVICE,
    )

    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={
            "device": settings.EMBEDDING_DEVICE,
        },
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": settings.EMBEDDING_BATCH_SIZE,
        },
    )