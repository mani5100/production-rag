from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "NXB Chatbot"
    APP_ENV: str = "development"

    CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 200

    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION_NAME: str = "nxb_docs"

    LLM_TEMPERATURE: float = 0.0
    MAX_TOKENS_TRIM: int = 4000
    RETRIEVER_TOP_K: int = 5
    RERANKER_TOP_N: int = 4

    DATA_FOLDER: str = "data"

    DATABASE_URL: str
    CHECKPOINTER_DATABASE_URL: str

    # LangSmith
    LANGSMITH_TRACING: str = "false"
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_PROJECT: str = "nxb-chatbot"

    TAVILY_API_KEY: str
    TAVILY_MAX_RESULTS: int = 2

    RERANK_SCORE_THRESHOLD: float = 0.3

    GMAIL_CLIENT_ID: str
    GMAIL_CLIENT_SECRET: str
    GMAIL_TOKEN_URI: str = "https://oauth2.googleapis.com/token"
    GMAIL_REFRESH_TOKEN: str
    GMAIL_SENDER_EMAIL: str
    MEAL_DEPARTMENT_EMAIL: str
    MIS_DEPARTMENT_EMAIL: str
    GMAIL_APP_PASSWORD: str = ""

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "qwen3:8b"
    LLM_MAX_TOKENS: int = 4096

    EMBEDDING_MODEL: str = "nomic-embed-text"
    EMBEDDING_DIMENSIONS: int = 768
    EMBEDDING_BATCH_SIZE: int = 16

    REDIS_URL: str = "redis://localhost:6379"
    SEMANTIC_CACHE_ENABLED: bool = True
    SEMANTIC_CACHE_INDEX_NAME: str = "nxb_semantic_cache"
    SEMANTIC_CACHE_THRESHOLD: float = 0.95
    SEMANTIC_CACHE_TTL: int = 86400
    
    RERANK_SCORE_THRESHOLD: float = 0.3
    SIMPLE_ROUTE_RELEVANCE_FLOOR: float = 0.3
    
    GM_EMAIL: str
    
    MAX_RETRIEVAL_ATTEMPTS: int = 2

    # Guardrail
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()