import os
import sys
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from nxb_chatbot.api.chat.router import router as chat_router
from nxb_chatbot.api.ingest.router import router as ingest_router

from nxb_chatbot.core.config import settings
from nxb_chatbot.core.startup import run_startup
from nxb_chatbot.rag.graph import get_compiled_graph

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


os.environ["LANGCHAIN_TRACING_V2"] = settings.LANGSMITH_TRACING
os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up NXB Chatbot...")

    await run_startup()

    graph, pool = await get_compiled_graph()
    app.state.graph = graph
    app.state.pool = pool

    logger.info("RAG graph ready.")
    yield

    # Shutdown
    logger.info("Shutting down NXB Chatbot...")
    await app.state.pool.close()
    logger.info("Postgres connection pool closed.")


app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api/v1")
app.include_router(ingest_router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "ok", "app": settings.APP_NAME}


@app.get("/")
async def health():
    return {"detail": "NextBridge Chatbot"}
