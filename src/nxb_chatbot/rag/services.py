import logging
import os

from langchain_core.messages import trim_messages
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from nxb_chatbot.rag.prompts import guardrail_prompt
from nxb_chatbot.core.config import settings
from nxb_chatbot.core.config import settings
from nxb_chatbot.rag.schema import GuardrailResult
from nxb_chatbot.rag.state import ChatState

logger = logging.getLogger(__name__)


# LLM instance
llm = ChatOpenAI(
    model=settings.LLM_MODEL,
    temperature=settings.LLM_TEMPERATURE,
    api_key=settings.OPENAI_API_KEY,
)



# Context Formatter

def format_context(state: ChatState) -> str:
    if not state["retrieved_docs"]:
        return "No relevant documents found."

    formatted = []
    for i, doc in enumerate(state["retrieved_docs"], 1):
        meta = doc.get("metadata", {})
        header = (
            f"[Document {i} - {meta.get('file_name', 'unknown')} | "
            f"Page {meta.get('page', '?')} | "
            f"Table: {meta.get('has_table', False)}]"
        )
        formatted.append(f"{header}\n{doc.get('page_content', '')}")

    return "\n\n---\n\n".join(formatted)



# Message Trimmer
def trim_conversation(state: ChatState) -> list:
    """
    Trim conversation history to MAX_TOKENS_TRIM.
    Always keeps most recent messages, never truncates mid-message.
    """
    return trim_messages(
        state["messages"],
        max_tokens=settings.MAX_TOKENS_TRIM,
        strategy="last",
        token_counter=llm,
        include_system=True,
        allow_partial=False,
    )


def get_tavily_search() -> TavilySearchResults:
    """
    Returns a Tavily search tool scoped to NextBridge related queries.
    Used as fallback when RAG retrieval score is below threshold.
    """
    os.environ["TAVILY_API_KEY"] = settings.TAVILY_API_KEY
    return TavilySearchResults(
        max_results=settings.TAVILY_MAX_RESULTS,
        api_key=settings.TAVILY_API_KEY,
    )
    
def get_guardrail_chain():
    """
    Returns a structured output chain for topic classification.
    Returns GuardrailResult with passed: bool and reason: str.
    """
    return guardrail_prompt | llm.with_structured_output(GuardrailResult)
