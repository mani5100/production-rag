# NXB Chatbot

A production-grade Retrieval-Augmented Generation (RAG) assistant that answers employee questions from company documents, handles HR-style workflows (meal subscriptions, MIS support, leave/WFH requests) over email, and supports both text and voice interaction.

## Features

- **Document RAG pipeline** — ingests PDFs from a local folder, splits them (tables kept atomic, text chunked), embeds them, and upserts them into a Qdrant vector store with duplicate/change detection.
- **LangGraph conversational agent** — a stateful graph (`src/nxb_chatbot/rag/graph.py`) that routes each turn through a guardrail, query reformulation, retrieval, reranking, web-search fallback, and answer generation.
- **Postgres-backed conversation memory** — uses `langgraph-checkpoint-postgres` to persist chat state per session, with session listing and message history endpoints.
- **HR workflow automation via Gmail** — dedicated LangGraph nodes and Gmail tools (`src/nxb_chatbot/tools/gmail.py`) for:
  - Meal subscription requests and acknowledgments
  - MIS (Management Information System) support requests
  - Employee leave / work-from-home requests routed to the GM
- **Web search fallback** — falls back to Tavily search when the internal knowledge base has no confident answer.
- **Reranking** — retrieved chunks are reranked (FlashRank) and filtered by a relevance score threshold before being passed to the LLM.
- **Voice interface** — separate speech-to-text (Faster-Whisper) and text-to-speech (Piper) FastAPI microservices.
- **Chainlit frontend** — a chat UI that talks to the backend API, supports microphone input, and streams TTS audio back over WebSocket.
- **Database migrations** — schema managed with Alembic.

## Tech Stack

- **Language:** Python 3.11
- **API framework:** FastAPI, Uvicorn
- **Agent / RAG orchestration:** LangGraph, LangChain (`langchain`, `langchain-community`, `langchain-openai`, `langchain-huggingface`, `langchain-pymupdf4llm`, `langchain-qdrant`, `langchain-text-splitters`)
- **LLM provider:** Groq (OpenAI-compatible API)
- **Embeddings:** Hugging Face (`BAAI/bge-base-en-v1.5` via `fastembed` / `sentence-transformers`)
- **Vector store:** Qdrant
- **Relational database:** PostgreSQL (via SQLAlchemy + `asyncpg`/`psycopg`, Alembic migrations)
- **Reranking:** FlashRank
- **Web search:** Tavily API
- **Email automation:** Gmail API (`google-api-python-client`, `google-auth`)
- **Speech-to-text:** Faster-Whisper
- **Text-to-speech:** Piper TTS
- **Frontend:** Chainlit
- **Package/dependency management:** `uv`
- **Containerization:** Docker, Docker Compose

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker and Docker Compose (recommended for running Postgres, Qdrant, and all services together)
- API keys for: Groq, OpenAI (embeddings), Tavily, and a Gmail OAuth app (client ID/secret + refresh token)

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd nxb-chatbot
```

Install dependencies with `uv`:

```bash
uv sync
```

Create your environment file from the template below (see [Configuration](#configuration)) and save it as `.env` in the project root.

### Running with Docker Compose (recommended)

This starts Postgres, Qdrant, the API, TTS, STT, and the Chainlit frontend together:

```bash
docker compose up --build
```

### Running locally without Docker

Start Postgres and Qdrant yourself (or via `docker compose up postgres qdrant`), then run the API:

```bash
uv run uvicorn nxb_chatbot.main:app --host 0.0.0.0 --port 8000
```

Apply database migrations:

```bash
uv run alembic upgrade head
```

## Configuration

All configuration is loaded from a `.env` file in the project root (see `src/nxb_chatbot/core/config.py`). Required and optional variables:

```env
# LLM (Groq)
GROQ_API_KEY=
GROQ_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-20b
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=4096

# Embeddings (OpenAI or Hugging Face, per config)
OPENAI_API_KEY=
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_DIMENSIONS=768
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=16

# Qdrant vector store
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=nxb_docs

# PostgreSQL (app database + LangGraph checkpointer)
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5433/nxb_chatbot
CHECKPOINTER_DATABASE_URL=postgresql://user:password@localhost:5433/nxb_chatbot

# Ingestion
DATA_FOLDER=data
CHUNK_SIZE=1024
CHUNK_OVERLAP=200

# Retrieval / reranking
RETRIEVER_TOP_K=5
RERANKER_TOP_N=4
RERANK_SCORE_THRESHOLD=0.3
MAX_TOKENS_TRIM=4000

# Web search fallback
TAVILY_API_KEY=
TAVILY_MAX_RESULTS=2

# Gmail (for meal/MIS/leave email workflows)
GMAIL_CLIENT_ID=
GMAIL_CLIENT_SECRET=
GMAIL_TOKEN_URI=https://oauth2.googleapis.com/token
GMAIL_REFRESH_TOKEN=
GMAIL_SENDER_EMAIL=
MEAL_DEPARTMENT_EMAIL=
MIS_DEPARTMENT_EMAIL=
GM_EMAIL=

# LangSmith (optional tracing)
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT=nxb-chatbot

# App
APP_NAME=NXB Chatbot
APP_ENV=development
```

To generate a Gmail OAuth refresh token, use the helper script:

```bash
uv run python scripts/generate_gmail_token.py
```

## Usage

### 1. Ingest documents

Place PDF files in the `data/` folder, then trigger ingestion via the API:

```bash
curl -X POST http://localhost:8000/api/v1/ingest/
```

### 2. Chat with the assistant

Send a message (non-streaming):

```bash
curl -X POST http://localhost:8000/api/v1/chat/ \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the meal subscription policy?"}'
```

Stream a response (NDJSON):

```bash
curl -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about the leave policy"}'
```

List past sessions:

```bash
curl http://localhost:8000/api/v1/chat/sessions
```

Get message history for a session:

```bash
curl http://localhost:8000/api/v1/chat/sessions/{session_id}/messages
```

### 3. Use the Chainlit frontend

If running via Docker Compose, the frontend is available at `http://localhost:8001`. It supports typed and voice input, and streams TTS audio replies.

### 4. Health check

```bash
curl http://localhost:8000/health
```

## Testing

Run the ingestion pipeline smoke test:

```bash
uv run python test_ingestion.py
```

Verify the Groq LLM connection:

```bash
uv run python test_rag.py
```

Run the automated test suite (if present under `tests/`):

```bash
uv run pytest
```

## Project Structure

```
nxb-chatbot/
├── alembic/                     # Database migration scripts
├── data/                        # Source PDFs for ingestion
├── scripts/
│   └── generate_gmail_token.py  # Gmail OAuth refresh-token helper
├── src/nxb_chatbot/
│   ├── main.py                  # FastAPI app entry point (lifespan, routers)
│   ├── api/
│   │   ├── chat/                # Chat endpoints, schemas, service logic
│   │   ├── ingest/               # Ingestion trigger endpoint
│   │   └── deps.py               # Shared FastAPI dependencies (DB session, graph)
│   ├── core/
│   │   ├── config.py             # Pydantic settings (env-driven)
│   │   ├── embeddings.py         # Embedding model setup
│   │   └── startup.py            # Startup routines
│   ├── db/                       # SQLAlchemy base and async session
│   ├── ingestion/
│   │   ├── loaders.py            # PDF loading
│   │   ├── splitters.py          # Chunking logic
│   │   └── pipeline.py           # End-to-end ingestion pipeline
│   ├── vector_store/
│   │   └── qdrant_client.py      # Qdrant upsert/query logic
│   ├── rag/
│   │   ├── graph.py              # LangGraph state graph definition
│   │   ├── nodes.py              # Graph node implementations
│   │   ├── prompts.py            # Prompt templates
│   │   ├── reranker.py           # FlashRank reranking
│   │   ├── services.py           # LLM client setup
│   │   └── state.py              # Graph state schema
│   ├── tools/
│   │   └── gmail.py              # Gmail-based HR workflow tools
│   ├── frontend/
│   │   └── app.py                # Chainlit chat UI
│   ├── stt/
│   │   └── app.py                # Speech-to-text microservice (Faster-Whisper)
│   └── tts/
│       ├── app.py                # Text-to-speech microservice (Piper)
│       └── voices/                # Piper voice models
├── docker-compose.yml             # Multi-service orchestration
├── Dockerfile                     # App container image
├── pyproject.toml                 # Project metadata and dependencies
└── alembic.ini                    # Alembic configuration
```
