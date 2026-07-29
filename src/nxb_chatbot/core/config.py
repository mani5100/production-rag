from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "NXB Chatbot"
    APP_ENV: str = "development"
    
    CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 200
    
    OPENAI_API_KEY: str
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS: int = 1536
    
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION_NAME: str = "nxb_docs"
    
    LLM_TEMPERATURE: float = 0.0
    MAX_TOKENS_TRIM: int = 4000
    RETRIEVER_TOP_K: int = 10
    RERANKER_TOP_N: int = 4
    
    DATA_FOLDER: str = "data"
    
    LLM_MODEL: str = "gpt-4o-mini"
    
    MAX_TOKENS_TRIM: int = 4000
    RETRIEVER_TOP_K: int = 5
    
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
    
    
    GROQ_API_KEY: str
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    LLM_MODEL: str = "openai/gpt-oss-20b"
    LLM_MAX_TOKENS: int = 4096
    
    
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    EMBEDDING_DIMENSIONS: int = 768
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_BATCH_SIZE: int = 16
    
    
    # Guardrail
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()