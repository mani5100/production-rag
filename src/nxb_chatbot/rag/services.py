from html import escape
import logging
import os

from langchain_core.messages import trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults
from nxb_chatbot.rag.prompts import guardrail_prompt, grade_prompt
from nxb_chatbot.core.config import settings
from nxb_chatbot.rag.schema import GuardrailResult, GradeDocuments
from nxb_chatbot.rag.state import ChatState

logger = logging.getLogger(__name__)


# # LLM instance
# llm = ChatOpenAI(
#     model=settings.LLM_MODEL,
#     temperature=settings.LLM_TEMPERATURE,
#     api_key=settings.OPENAI_API_KEY,
# )

# LLM instance
llm = ChatOllama(
    model=settings.LLM_MODEL,
    temperature=settings.LLM_TEMPERATURE,
    base_url=settings.OLLAMA_BASE_URL,
    num_ctx=8192,
    num_predict=settings.LLM_MAX_TOKENS,
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


def trim_conversation(state: ChatState) -> list:
    """
    Trim conversation history without depending on provider-specific
    tokenizer support.
    """
    return trim_messages(
        state["messages"],
        max_tokens=settings.MAX_TOKENS_TRIM,
        strategy="last",
        token_counter=count_tokens_approximately,
        include_system=True,
        allow_partial=False,
    )


def get_tavily_search() -> TavilySearchResults:
    """
    Returns a Tavily search tool scoped to NextBridge related queries.
    Used as fallback when RAG retrieval score is below threshold.

    Restricted to nextbridge.com so results can't come from unrelated
    companies or generic pages that happen to match the query text.
    """
    os.environ["TAVILY_API_KEY"] = settings.TAVILY_API_KEY
    return TavilySearchResults(
        max_results=settings.TAVILY_MAX_RESULTS,
        api_key=settings.TAVILY_API_KEY
    )


def get_guardrail_chain():
    """
    Returns a structured output chain for topic classification.
    Returns GuardrailResult with passed: bool and reason: str.
    """
    return guardrail_prompt | llm.with_structured_output(GuardrailResult)


def get_grading_chain():
    """
    Returns a structured output chain for CRAG document grading.
    Returns GradeDocuments with verdict: 'relevant' | 'irrelevant' and reason: str.
    """
    return grade_prompt | llm.with_structured_output(GradeDocuments)


def _plain_text_to_html(value: str) -> str:
    """
    Escapes LLM-generated email text before placing it into an HTML message.
    """
    return escape(value).replace("\n", "<br>")


def _merge_non_null_values(
    current: dict,
    updates: dict,
    allowed_fields: set[str],
) -> dict:
    """
    Merge only non-null LLM-extracted fields into workflow state.
    """
    merged = dict(current)

    for key in allowed_fields:
        value = updates.get(key)

        if value is not None:
            if isinstance(value, str):
                value = value.strip()

            if value != "":
                merged[key] = value

    return merged


def _extract_tracking_data(result: str) -> tuple[str | None, str | None]:
    thread_id: str | None = None
    request_reference: str | None = None

    if "thread_id=" in result:
        thread_id = result.split("thread_id=", 1)[1].split(";", 1)[0].strip() or None

        if thread_id and thread_id.lower() == "none":
            thread_id = None

    if "request_reference=" in result:
        request_reference = (
            result.split("request_reference=", 1)[1].split(";", 1)[0].strip() or None
        )

    return thread_id, request_reference


def _employee_request_view(request_data: dict) -> dict:
    """
    Only expose meaningful request fields to the LLM.
    Avoid passing internal flags as if they were employee-provided facts.
    """
    keys = {
        "request_type",
        "employee_name",
        "employee_id",
        "start_date",
        "end_date",
        "reason",
        "confirmation_requested",
        "email_sent",
    }

    return {
        key: request_data.get(key) for key in keys if request_data.get(key) is not None
    }
