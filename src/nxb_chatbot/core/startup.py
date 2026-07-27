import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from alembic import command
from alembic.config import Config

from nxb_chatbot.vector_store.qdrant_client import init_collection

logger = logging.getLogger(__name__)


def _run_migrations_sync() -> None:
    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")


async def run_migrations() -> None:
    try:
        logger.info("Running database migrations...")
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            await loop.run_in_executor(pool, _run_migrations_sync)
        logger.info("Database migrations complete.")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise


def setup_qdrant() -> None:
    try:
        logger.info("Setting up Qdrant collection...")
        init_collection()
        logger.info("Qdrant collection ready.")
    except Exception as e:
        logger.error(f"Qdrant setup failed: {e}")
        raise


async def run_startup() -> None:
    await run_migrations()
    setup_qdrant()