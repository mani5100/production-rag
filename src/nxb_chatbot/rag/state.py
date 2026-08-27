from typing import Annotated

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

    retrieved_docs: list[dict]
    retrieval_filters: dict | None

    standalone_query: str | None
    retrieval_queries: list[str] | None

    generated_answer: str | None

    guardrail_passed: bool | None
    web_search_used: bool | None

    retrieval_attempts: int
    grade_verdict: str | None
    grade_reason: str | None

    reflection_action: str | None
    reflection_reason: str | None
    reflection_feedback: str | None

    generation_attempts: int
    reflection_attempts: int