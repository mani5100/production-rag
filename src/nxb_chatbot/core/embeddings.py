import logging

from langchain_openai import OpenAIEmbeddings

from nxb_chatbot.core.config import settings

logger = logging.getLogger(__name__)


def get_dense_embedder() -> OpenAIEmbeddings:
    """
    Returns the dense embedder using OpenAI text-embedding-3-small.
    This is used by both the ingestion pipeline and the RAG retriever.
    """
    return OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        dimensions=settings.EMBEDDING_DIMENSIONS,
        api_key=settings.OPENAI_API_KEY,
    )